//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX_Point.hpp>
#include <ArborX_Triangle.hpp>
#include <Kokkos_Core.hpp>
#include <limits>

#include "common/types.hpp"
#include "finite_elements/utils/ArborXCallbacks.hpp"

namespace PACMAN {
namespace FiniteElements {

/**
 * @brief Compute the closest point on a 2D edge segment.
 * @param[in] p Query point.
 * @param[in] a First segment endpoint.
 * @param[in] b Second segment endpoint.
 * @return Closest point to `p` on segment `[a,b]`.
 */
KOKKOS_INLINE_FUNCTION ArborX::Point<2, coordinates_t>
closest_point_to_edge(const ArborX::Point<2, coordinates_t> &p,
                      const ArborX::Point<2, coordinates_t> &a,
                      const ArborX::Point<2, coordinates_t> &b) {
  const auto ab = b - a;
  const auto ap = p - a;
  coordinates_t t =
      (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]);
  auto tclamp = Kokkos::max(0.0, Kokkos::min(1.0, t));
  ArborX::Point<2, coordinates_t> ret = {a[0] + ab[0] * tclamp,
                                         a[1] + ab[1] * tclamp};
  return ret;
};

/**
 * @brief Project outside target points on a 3D skin and evaluate FE values.
 * @tparam ExecSpace Kokkos execution space used for queries and kernels.
 * @param[in,out] transfer Transfer descriptor containing mesh/field data.
 * Reads: source mesh/values, connectivity, skin entities, target points/status.
 * Writes: `targetValues`, `targetStatus`, and optionally projected
 * `targetPoints` when `extrapol == false`.
 * @param[in] extrapol If `true`, keep target coordinates and mark outside
 * points as `TransferStatus::EXTRAP`; otherwise clamp projected coordinates and
 * mark as `TransferStatus::CLAMP`.
 */
template <typename ExecSpace>
void ComputeProjectionOn3DSkin(Transfer<ExecSpace, 3> &transfer,
                               bool extrapol = false) {
  Kokkos::Profiling::pushRegion("FiniteElements::ComputeProjectionOn3DSkin");
  using MemorySpace = typename ExecSpace::memory_space;
  using Triangle = ArborX::Triangle<3, coordinates_t>;
  using Point = ArborX::Point<3, coordinates_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;
  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto CellTypesPtr = transfer.cellTypes;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent_int(0);

  auto skinFacesPtr = transfer.skinFaces;
  auto skinParentsPtr = transfer.skinParents;

  auto nbTri = transfer.skinFaces.extent_int(0);
  Kokkos::View<Triangle *, MemorySpace> skinFacesView(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Skin ArborX Triangles"),
      nbTri);
  Kokkos::parallel_for(
      "Fill ArborX triangles",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbTri),
      KOKKOS_LAMBDA(const int_t &i) {
        for (int_t d = 0; d < 3; d++) {
          skinFacesView(i).a[d] = sourcePointsPtr(skinFacesPtr(i, 0))[d];
          skinFacesView(i).b[d] = sourcePointsPtr(skinFacesPtr(i, 1))[d];
          skinFacesView(i).c[d] = sourcePointsPtr(skinFacesPtr(i, 2))[d];
        }
      });
  ArborX::BoundingVolumeHierarchy skinBVH(
      execSpace, ArborX::Experimental::attach_indices(skinFacesView));
  Kokkos::Profiling::popRegion(); // Build triangle BoundingVolumeHierarchy

  Kokkos::Profiling::pushRegion("Compute query of nearest skin");

  Kokkos::View<int_t *, MemorySpace> values("values", 0);
  Kokkos::View<offset_t *, MemorySpace> offsets("offsets", 0);

  auto targetElement = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Target to elem"),
      nbtargetPoints);

  Kokkos::Profiling::pushRegion("Point projection on triangle");
  PointCloudNearest<MemorySpace, 3> pcn{targetPointsPtr};
  if (extrapol) {
    skinBVH.query(execSpace, pcn,
                  PointTriangleProjectionExtrapol<ExecSpace, 3>{
                      targetElement, targetStatusPtr, skinParentsPtr},
                  values, offsets);
  } else {
    skinBVH.query(
        execSpace, pcn,
        PointTriangleProjection<ExecSpace, 3>{targetPointsPtr, targetElement,
                                              targetStatusPtr, skinParentsPtr},
        values, offsets);
  }
  Kokkos::Profiling::popRegion(); // Point projection on triangle

  auto statusOutside =
      (extrapol) ? TransferStatus::EXTRAP : TransferStatus::CLAMP;

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Compute projection values ",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, values.extent_int(0)),
      KOKKOS_LAMBDA(const int_t &i) {
        auto predicate_index = values(i);
        Kokkos::Array<fp_t, MaxNodesPerElt> warr;
        Kokkos::Array<coordinates_t, MaxNodesPerElt * 3> Xarr;
        Kokkos::Array<coordinates_t, 3> tpa;
        Kokkos::View<coordinates_t *, ExecSpace> tp(tpa.data(), 3);
        for (int_t j = 0; j < 3; j++) {
          tp(j) = targetPointsPtr(predicate_index, j);
        }
        // predicate = target point data
        // primitive intersected bounding box finite element data
        const int_t curElem = targetElement(predicate_index);
        const auto cellType = CellTypesPtr(curElem);
        auto localConn = Kokkos::subview(
            connValPtr,
            Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
        int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
        Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
        Kokkos::View<coordinates_t **, ExecSpace> Xcoor(Xarr.data(),
                                                        nbConnNodes, 3);
        for (int_t j = 0; j < nbConnNodes; ++j) {
          auto nodeId = localConn(j);
          for (int_t k = 0; k < 3; ++k) {
            Xcoor(j, k) = sourcePointsPtr(nodeId)[k];
          }
        }
        auto isInside = ApplyNewtonOnElement<ExecSpace, 3>(cellType, Xcoor, tp,
                                                           weights, true);
        for (int_t j = 0; j < nbConnNodes; ++j) {
          auto nodeId = localConn(j);
          targetValuesPtr(predicate_index) +=
              weights[j] * sourceValuesPtr(nodeId);
        }
        targetStatusPtr(predicate_index) = statusOutside;
      });
  Kokkos::Profiling::popRegion(); // Compute projection and interpolation
                                  // coefficient
};

/**
 * @brief Project outside target points on a 2D skin and evaluate FE values.
 * @tparam ExecSpace Kokkos execution space used for kernels.
 * @param[in,out] transfer Transfer descriptor containing mesh/field data.
 * Reads: source mesh/values, connectivity, skin entities, target points/status.
 * Writes: `targetValues`, `targetStatus`, and optionally projected
 * `targetPoints` when `extrapol == false`.
 * @param[in] extrapol If `true`, keep target coordinates and mark outside
 * points as `TransferStatus::EXTRAP`; otherwise clamp projected coordinates and
 * mark as `TransferStatus::CLAMP`.
 */
template <typename ExecSpace>
void ComputeProjectionOn2DSkin(Transfer<ExecSpace, 2> &transfer,
                               bool extrapol = false) {
  Kokkos::Profiling::pushRegion("FiniteElements::ComputeProjectionOn2DSkin");
  using MemorySpace = typename ExecSpace::memory_space;
  using Point = ArborX::Point<2, coordinates_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;
  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto CellTypesPtr = transfer.cellTypes;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent_int(0);

  auto skinFacesPtr = transfer.skinFaces;
  auto skinParentsPtr = transfer.skinParents;

  auto statusOutside =
      (extrapol) ? TransferStatus::EXTRAP : TransferStatus::CLAMP;

  auto targetElement = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Target to elem"),
      nbtargetPoints);

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Retrieve closest bar",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0,
                                     targetPointsPtr.extent_int(0)),
      KOKKOS_LAMBDA(const int_t &i) {
        if (targetStatusPtr(i) == TransferStatus::OUTSIDE) {
          Kokkos::Array<fp_t, MaxNodesPerElt> warr;
          Kokkos::Array<coordinates_t, MaxNodesPerElt * 2> Xarr;
          Kokkos::Array<coordinates_t, 2> tpa;
          Kokkos::View<coordinates_t *, ExecSpace> tp(tpa.data(), 2);
          const Point target = {targetPointsPtr(i, 0), targetPointsPtr(i, 1)};
          fp_t mindistsqr = fp_consts::max();
          int_t closestId = -1;
          Point closest;
          for (int_t s = 0; s < skinFacesPtr.extent_int(0); s++) {
            auto p1 = sourcePointsPtr(skinFacesPtr(s, 0));
            auto p2 = sourcePointsPtr(skinFacesPtr(s, 1));
            auto cur = closest_point_to_edge(target, p1, p2);
            coordinates_t dx = cur[0] - target[0];
            coordinates_t dy = cur[1] - target[1];
            coordinates_t curdistsqr = dx * dx + dy * dy;
            if (curdistsqr < mindistsqr) {
              closest = cur;
              mindistsqr = curdistsqr;
              closestId = s;
            }
          }
          if (!extrapol) {
            targetPointsPtr(i, 0) = closest[0];
            targetPointsPtr(i, 1) = closest[1];
          }
          for (int_t j = 0; j < 2; j++) {
            tp(j) = targetPointsPtr(i, j);
          }
          targetElement(i) = skinParentsPtr(closestId);
          const int_t curElem = targetElement(i);
          const auto cellType = CellTypesPtr(curElem);
          auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          const int_t nbConnNodes =
              connOffPtr(curElem + 1) - connOffPtr(curElem);
          Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
          Kokkos::View<coordinates_t **, ExecSpace> Xcoor(Xarr.data(),
                                                          nbConnNodes, 2);
          for (int_t j = 0; j < nbConnNodes; ++j) {
            const auto nodeId = localConn(j);
            for (int_t k = 0; k < 2; ++k) {
              Xcoor(j, k) = sourcePointsPtr(nodeId)[k];
            }
          }
          auto isInside = FiniteElements::ApplyNewtonOnElement<ExecSpace, 2>(
              cellType, Xcoor, tp, weights, true);
          for (int_t j = 0; j < nbConnNodes; ++j) {
            const auto nodeId = localConn(j);
            targetValuesPtr(i) += weights[j] * sourceValuesPtr(nodeId);
          }
          targetStatusPtr(i) = statusOutside;
        }
      });
  Kokkos::Profiling::popRegion(); // Compute projection and interpolation
                                  // coefficient
};

/**
 * @brief Project outside target points on a 1D skin and evaluate FE values.
 * @tparam ExecSpace Kokkos execution space used for kernels.
 * @param[in,out] transfer Transfer descriptor containing mesh/field data.
 * Reads: source mesh/values, connectivity, skin entities, target points/status.
 * Writes: `targetValues`, `targetStatus`, and optionally projected
 * `targetPoints` when `extrapol == false`.
 * @param[in] extrapol If `true`, keep target coordinates and mark outside
 * points as `TransferStatus::EXTRAP`; otherwise clamp projected coordinates and
 * mark as `TransferStatus::CLAMP`.
 */
template <typename ExecSpace>
void ComputeProjectionOn1DSkin(Transfer<ExecSpace, 1> &transfer,
                               bool extrapol = false) {
  Kokkos::Profiling::pushRegion("FiniteElement::ComputeProjectionOn1DSkin");
  using MemorySpace = typename ExecSpace::memory_space;
  using Point = ArborX::Point<1, coordinates_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;
  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto CellTypesPtr = transfer.cellTypes;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent_int(0);

  auto skinFacesPtr = transfer.skinFaces;
  auto skinParentsPtr = transfer.skinParents;

  auto statusOutside =
      (extrapol) ? TransferStatus::EXTRAP : TransferStatus::CLAMP;

  auto targetElement = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Target to elem"),
      nbtargetPoints);

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Retrieve closest point",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0,
                                     targetPointsPtr.extent_int(0)),
      KOKKOS_LAMBDA(const int &i) {
        if (targetStatusPtr(i) == TransferStatus::OUTSIDE) {
          Kokkos::Array<fp_t, MaxNodesPerElt> warr;
          Kokkos::Array<coordinates_t, MaxNodesPerElt> Xarr;
          Kokkos::Array<coordinates_t, 1> tpa;
          Kokkos::View<coordinates_t *, ExecSpace> tp(tpa.data(), 1);
          const Point target = {targetPointsPtr(i, 0)};
          fp_t mindist = fp_consts::max();
          int_t closestId = -1;
          Point closest;
          for (int_t s = 0; s < skinFacesPtr.extent_int(0); s++) {
            auto p = sourcePointsPtr(skinFacesPtr(s, 0));
            const auto curdist = Kokkos::abs(p[0] - target[0]);
            if (curdist < mindist) {
              closest = p;
              mindist = curdist;
              closestId = s;
            }
          }
          if (!extrapol) {
            targetPointsPtr(i, 0) = closest[0];
          }
          for (int_t j = 0; j < 1; j++) {
            tp(j) = targetPointsPtr(i, j);
          }
          targetElement(i) = skinParentsPtr(closestId);
          const int_t curElem = targetElement(i);
          const auto cellType = CellTypesPtr(curElem);
          auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
          Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
          Kokkos::View<coordinates_t **, ExecSpace> Xcoor(Xarr.data(),
                                                          nbConnNodes, 1);
          for (int_t j = 0; j < nbConnNodes; ++j) {
            const auto nodeId = localConn(j);
            for (int_t k = 0; k < 1; ++k) {
              Xcoor(j, k) = sourcePointsPtr(nodeId)[k];
            }
          }
          auto isInside = ApplyNewtonOnElement<ExecSpace, 1>(cellType, Xcoor,
                                                             tp, weights, true);
          for (int_t j = 0; j < nbConnNodes; ++j) {
            const auto nodeId = localConn(j);
            targetValuesPtr(i) += weights[j] * sourceValuesPtr(nodeId);
          }
          targetStatusPtr(i) = statusOutside;
        }
      });
  Kokkos::Profiling::popRegion(); // Compute projection and interpolation
                                  // coefficient
};

} // namespace FiniteElements
} // namespace PACMAN
