#pragma once

#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>

#include "common/transfer.hxx"
#include "common/types.hpp"
#include "finite_elements/utils/ArborXCallbacks.hpp"

namespace PACMAN {
namespace FiniteElements {
/**
 * @brief Compute the nearest point in the mesh for all target points and
 * assigns its value.
 * @param[in] transfer the transfer class holding informations.
 * @return target point values
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
  Kokkos::fence();
  Kokkos::Profiling::popRegion();
}

} // namespace FiniteElements

} // namespace PACMAN
