#pragma once

#include <Kokkos_Abort.hpp>
#include <stdexcept>

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FETools.hxx"
#include "finite_elements/utils/NewtonKokkos.hpp"

#define CaseSwitch(CellType_STRUCT)                                            \
  case CellType_STRUCT: {                                                      \
    return ApplyNewton<ExecSpace,                                              \
                       LagrangeSpaceGeo<ExecSpace, CellType_STRUCT>,           \
                       spaceDimension>(Xcoor, targetPoint, weights,            \
                                       forceEvaluation);                       \
    break;                                                                     \
  }

namespace PACMAN {

namespace FiniteElements {

template <typename ExecSpace, unsigned int spaceDimension>
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
    const int_t type, const Kokkos::View<fp_t **, ExecSpace> Xcoor,
    const Kokkos::View<fp_t[spaceDimension], ExecSpace> targetPoint,
    Kokkos::View<fp_t *, ExecSpace> weights,
    const bool forceEvaluation = false) {
  // clang-format off
            switch (static_cast<CellType>(type))
            {
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
            return false;
  // clang-format on
}

template <typename ExecSpace, int spaceDimension>
void ComputeBoxTargetPointIntersection(
    Transfer<ExecSpace, spaceDimension> &transfer) {
  using MemorySpace = typename ExecSpace::memory_space;
  using Box = ArborX::Box<spaceDimension, fp_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent(0);

  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto elementsTypePtr = transfer.cellTypes;

  Kokkos::View<Box *, MemorySpace> bboxes(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Boundary boxes of elements"),
      transfer.nbElems);
  Kokkos::parallel_for(
      "Compute element bounding boxes functor",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, transfer.nbElems),
      KOKKOS_LAMBDA(int i) {
        bboxes(i) = Box();
        auto localConn = Kokkos::subview(
            connValPtr, Kokkos::make_pair(connOffPtr(i), connOffPtr(i + 1)));
        int nbConnNodes = connOffPtr(i + 1) - connOffPtr(i);
        for (int j = 0; j < nbConnNodes; j++) {
          ArborX::Details::expand(bboxes(i), sourcePointsPtr(localConn(j)));
        }
      });
  ArborX::BoundingVolumeHierarchy bvhBoxes(
      execSpace, ArborX::Experimental::attach_indices(bboxes));
  auto intersectValues = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "IntersectValues"),
      0);
  auto intersectOffsets = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "IntersectOffsets"),
      0);

  PointIntersect<MemorySpace, spaceDimension> pi{targetPointsPtr};
  bvhBoxes.query(execSpace, pi, ExtractIndex{}, intersectValues,
                 intersectOffsets);

  // TODO: Use hierachical //ism and fefine scratch memory with
  // level 0 : tp
  // level 1 : weights, Xcoor
  Kokkos::parallel_for(
      "Compute Point Elt Inteserction and Interpole",
      Kokkos::RangePolicy(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(const int i) {
        Kokkos::Array<fp_t, MaxNodesPerElt> warr;
        Kokkos::Array<fp_t, MaxNodesPerElt * spaceDimension> Xarr;
        Kokkos::View<fp_t[spaceDimension], ExecSpace> tp;
        for (int j = 0; j < spaceDimension; j++) {
          tp[j] = targetPointsPtr(i, j);
        }
        for (int ibox = intersectOffsets(i); ibox < intersectOffsets(i + 1);
             ibox++) {
          // predicate = target point data
          // primitive intersected bounding box finite element data
          const int_t curElem = intersectValues(ibox);
          const auto elemType = elementsTypePtr(curElem);
          auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
          Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
          Kokkos::View<fp_t **, ExecSpace> Xcoor(Xarr.data(), nbConnNodes,
                                                 spaceDimension);
          for (int j = 0; j < nbConnNodes; ++j) {
            auto nodeId = localConn(j);
            for (int k = 0; k < spaceDimension; ++k) {
              Xcoor(j, k) = sourcePointsPtr(nodeId)[k];
            }
          }
          if (ApplyNewtonOnElement<ExecSpace, spaceDimension>(
                  elemType, Xcoor, tp, weights /*, false*/)) {
            for (int j = 0; j < nbConnNodes; ++j) {
              auto nodeId = localConn(j);
              targetValuesPtr(i) = weights[j] * sourceValuesPtr(nodeId);
            }
            targetStatusPtr(i) = static_cast<status_t>(TransferStatus::INTER);
            return;
          }
        }
      });
  Kokkos::fence();
  return;
}

} // namespace FiniteElements
} // namespace PACMAN
