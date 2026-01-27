#pragma once

#include <Kokkos_Abort.hpp>
#include <stdexcept>

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FETools.hxx"
#include "finite_elements/utils/NewtonKokkos.hpp"

#define CaseSwitch(CellType_STRUCT)                                            \
  case CellType_STRUCT: {                                                      \
    return ApplyNewton<ExecSpace,                                              \
                       LagrangeSpaceGeo<ExecSpace, CellType_STRUCT>, Dim>(     \
        Xcoor, targetPoint, weights, forceEvaluation);                         \
    break;                                                                     \
  }

namespace PACMAN {

namespace FiniteElements {

template <typename ExecSpace, int_t Dim>
/**
 * @brief Redirector function to call the ApplyNewton with the right templates.
 * @param[in] type The element type specifying the template to use.
 * @param[in] xcoor The coordinates of the element.
 * @param[in] targetPoint The target point we want the barycentric coordinates.
 * @param[out] weights Interpolation weights
 * @param[in] forceEvaluation Force weights evaluation inside .
 * @return true if the target point lands inside of the element and the result
 * is not clamped.
 */
KOKKOS_FUNCTION bool ApplyNewtonOnElement(
    const cell_t type, const Kokkos::View<coordinates_t **, ExecSpace> Xcoor,
    const Kokkos::View<coordinates_t[Dim], ExecSpace> targetPoint,
    Kokkos::View<fp_t *, ExecSpace> weights,
    const bool forceEvaluation = false) {
  // clang-format off
  switch (static_cast<CellType>(type)) {
    CaseSwitch(CellType::VTK_LINE)
    CaseSwitch(CellType::VTK_QUADRATIC_HEXAHEDRON)
    CaseSwitch(CellType::VTK_QUADRATIC_EDGE)
    CaseSwitch(CellType::VTK_HEXAHEDRON)
    CaseSwitch(CellType::VTK_QUADRATIC_PYRAMID)
    CaseSwitch(CellType::VTK_PYRAMID)
    CaseSwitch(CellType::VTK_QUAD)
    CaseSwitch(CellType::VTK_QUADRATIC_QUAD)
    CaseSwitch(CellType::VTK_QUADRATIC_TETRA)
    CaseSwitch(CellType::VTK_TETRA)
    CaseSwitch(CellType::VTK_TRIANGLE)
    CaseSwitch(CellType::VTK_QUADRATIC_TRIANGLE)
    CaseSwitch(CellType::VTK_QUADRATIC_WEDGE)
    CaseSwitch(CellType::VTK_WEDGE)
    default:
      char errorMsg[] = "Unexpected element type XXX";
      errorMsg[sizeof(errorMsg) - 4] = '0' + ((int)type / 100);
      errorMsg[sizeof(errorMsg) - 3] = '0' + (((int)type / 10) % 10);
      errorMsg[sizeof(errorMsg) - 2] = '0' + ((int)type % 10);
      Kokkos::abort(errorMsg);
      return false;
  }
  // clang-format on
  return false;
}

template <typename ExecSpace, int_t Dim>
void ComputeBoxTargetPointIntersection(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion(
      "FiniteElement::ComputeBoxTargetPointIntersection");
  if (transfer.nbSpaceDimElements == 0) {
    return;
  }

  using MemorySpace = typename ExecSpace::memory_space;
  using Box = ArborX::Box<Dim, coordinates_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent_int(0);

  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto CellTypesPtr = transfer.cellTypes;

  Kokkos::View<Box *, MemorySpace> bboxes(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Boundary boxes of elements"),
      transfer.nbSpaceDimElements);
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Parents of Bboxes"),
      transfer.nbSpaceDimElements);

  Kokkos::parallel_scan(
      "Compute element bounding boxes functor",
      Kokkos::RangePolicy(execSpace, 0, transfer.nbElems),
      KOKKOS_LAMBDA(const int_t &i, int_t &partial_sum, bool is_final) {
        const CellType cellType = static_cast<CellType>(CellTypesPtr(i));
        const int_t elemToTreat =
            static_cast<int_t>(getDimension(cellType) == Dim);
        const int_t firstIndex = partial_sum;
        partial_sum += elemToTreat;
        if (is_final) {
          for (int_t j = firstIndex; j < firstIndex + elemToTreat; j++) {
            bboxes(j) = Box();
            auto localConn = Kokkos::subview(
                connValPtr,
                Kokkos::make_pair(connOffPtr(i), connOffPtr(i + 1)));
            const int_t nbConnNodes = connOffPtr(i + 1) - connOffPtr(i);
            for (int_t k = 0; k < nbConnNodes; k++) {
              ArborX::Details::expand(bboxes(j), sourcePointsPtr(localConn(k)));
            }
            parents(j) = i;
          }
        }
      });

  ArborX::BoundingVolumeHierarchy bvhBoxes(
      execSpace, ArborX::Experimental::attach_indices(bboxes));
  auto intersectValues = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "IntersectValues"),
      0);
  auto intersectOffsets = Kokkos::View<offset_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "IntersectOffsets"),
      0);

  PointIntersect<MemorySpace, Dim> pi{targetPointsPtr};
  bvhBoxes.query(execSpace, pi, ExtractIndex{}, intersectValues,
                 intersectOffsets);

  // TODO: Use hierachical //ism and define scratch memory with
  // level 0 : tp
  // level 1 : weights, Xcoor
  Kokkos::parallel_for(
      "Compute Point Elt Inteserction and Interpole",
      Kokkos::RangePolicy(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(const int_t &i) {
        Kokkos::Array<fp_t, MaxNodesPerElt> warr;
        Kokkos::Array<coordinates_t, MaxNodesPerElt * Dim> Xarr;
        Kokkos::Array<coordinates_t, Dim> tpa;
        Kokkos::View<coordinates_t *, ExecSpace> tp(tpa.data(), Dim);
        for (int_t j = 0; j < Dim; j++) {
          tp(j) = targetPointsPtr(i, j);
        }
        for (offset_t ibox = intersectOffsets(i);
             ibox < intersectOffsets(i + 1); ibox++) {
          // predicate = target point data
          // primitive intersected bounding box finite element data
          const int_t curElem = parents(intersectValues(ibox));
          const auto cellType = CellTypesPtr(curElem);
          const auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          const offset_t nbConnNodes =
              connOffPtr(curElem + 1) - connOffPtr(curElem);
          Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
          Kokkos::View<coordinates_t **, ExecSpace> Xcoor(Xarr.data(),
                                                          nbConnNodes, Dim);
          for (int_t j = 0; j < nbConnNodes; ++j) {
            const auto nodeId = localConn(j);
            for (int_t k = 0; k < Dim; ++k) {
              Xcoor(j, k) = sourcePointsPtr(nodeId)[k];
            }
          }
          if (ApplyNewtonOnElement<ExecSpace, Dim>(cellType, Xcoor, tp,
                                                   weights /*, false*/)) {
            for (int_t j = 0; j < nbConnNodes; ++j) {
              const auto nodeId = localConn(j);
              targetValuesPtr(i) += weights[j] * sourceValuesPtr(nodeId);
            }
            targetStatusPtr(i) = TransferStatus::INTER;
            return;
          }
        }
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // ComputeBoxTargetPointIntersection
}

} // namespace FiniteElements
} // namespace PACMAN
