//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "common/transfer.hxx"
#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FTUtils.hpp"

namespace PACMAN {
namespace FiniteElements {

/**
 * @brief Interpolate target points from FE cells and fallback to nearest
 * source-node value outside the source mesh.
 * @tparam ExecSpace Kokkos execution space used for search and interpolation.
 * @tparam Dim Spatial dimension of the interpolation problem.
 * @param[in,out] transfer Transfer descriptor holding interpolation data.
 * Reads: source mesh/field data and target points.
 * Writes: `targetValues` and `targetStatus`.
 * @note Target points not intersecting any source element are assigned the
 * value of their nearest source point and marked with
 * `TransferStatus::NEAREST`.
 */
template <typename ExecSpace, int_t Dim>
void FTInterpNearest(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion("FTInterpNearest");

  using MemorySpace = typename ExecSpace::memory_space;
  using Box = ::ArborX::Box<Dim, coordinates_t>;

  ExecSpace execSpace{};

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent(0);

  ::ArborX::BoundingVolumeHierarchy bvhPoints(
      execSpace, ::ArborX::Experimental::attach_indices(sourcePointsPtr));
  Kokkos::Profiling::pushRegion("Compute nearest points");
  auto nearestPointValues = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "nearestPointValues"),
      0);
  auto nearestPointOffsets = Kokkos::View<int_t *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "nearestPointOffsets"),
      0);
  PointCloudNearest<MemorySpace, Dim> pcn{targetPointsPtr};
  bvhPoints.query(execSpace, pcn, ExtractIndex{}, nearestPointValues,
                  nearestPointOffsets);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("Compute target point FE intersection");
  ComputeBoxTargetPointIntersection<ExecSpace, Dim>(transfer);
  Kokkos::Profiling::popRegion();

  // Assign nearest node source value for target points outside the mesh
  Kokkos::Profiling::pushRegion("Get Nearest");
  Kokkos::parallel_for(
      "Get Nearest functor",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(const int_t &i) {
        if (targetStatusPtr(i) != TransferStatus::INTER) {
          auto nearest_idx = nearestPointValues(i);
          targetValuesPtr(i) = sourceValuesPtr(nearest_idx);
          targetStatusPtr(i) = TransferStatus::NEAREST;
        }
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
  return;
}
} // namespace FiniteElements
} // namespace PACMAN
