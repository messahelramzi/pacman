//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>

#include "common/transfer.hxx"
#include "common/types.hpp"
#include "finite_elements/utils/ArborXCallbacks.hpp"

namespace PACMAN {
namespace FiniteElements {
/**
 * @brief Transfer source values to target points using nearest-neighbor search.
 * @tparam ExecSpace Kokkos execution space used for BVH build and query.
 * @tparam Dim Spatial dimension of the point clouds (1, 2, or 3).
 * @param[in,out] transfer Transfer descriptor holding interpolation data.
 * Reads: `sourcePoints`, `sourceValues`, `targetPoints`.
 * Writes: `targetValues`.
 * @note This method is point-based and does not use connectivity or cell
 * types.
 */
template <typename ExecSpace, int_t Dim>
void FTNearest(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion("FTNearest");

  using TransferType = Transfer<ExecSpace, Dim>;
  using MemorySpace = typename TransferType::MemorySpace;
  using Point = ::ArborX::Point<Dim, coordinates_t>;

  ExecSpace execspace{};

  auto source_points = transfer.sourcePoints;
  auto source_values = transfer.sourceValues;
  auto target_points = transfer.targetPoints;
  auto target_values = transfer.targetValues;

  auto nb_target_points = target_points.extent(0);

  Kokkos::Profiling::pushRegion("BVH construction and NN search");

  ::ArborX::BoundingVolumeHierarchy bvh(
      execspace, ::ArborX::Experimental::attach_indices(source_points));
  PointCloudNearest<MemorySpace, Dim> pcn{target_points};
  bvh.query(execspace, pcn,
            NearestExtractIndex<MemorySpace>{source_values, target_values});
  Kokkos::Profiling::popRegion();
}

} // namespace FiniteElements

} // namespace PACMAN
