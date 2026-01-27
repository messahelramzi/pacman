//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

#pragma once

#include "common/transfer.hxx"
#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FTUtils.hpp"

namespace PACMAN {
namespace FiniteElements {

template <typename ExecSpace, int_t Dim>
/**
 * @brief Compute the target point finite element interpolation if intersects or
 * assign nearest node value if outside all elements.
 * @param[in] transfer the transfer class holding informations.
 * @return target point values
 */
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
  Kokkos::fence();
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
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
  return;
}
} // namespace FiniteElements
} // namespace PACMAN
