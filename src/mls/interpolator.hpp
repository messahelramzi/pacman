//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <iostream>
#include <ArborX_InterpMovingLeastSquares.hpp>
#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"
#include "mls/interpolator.hxx"

namespace PACMAN {
namespace MLS {

/// Convert a 2-D `[N × Dim]` coordinate view into a 1-D view of
/// `ArborX::Point<Dim, coordinates_t>`.
template <KokkosViewRank<2> SourceType, KokkosViewRank<1> TargetType,
          int_t Dim>
static inline void FillPoints(const SourceType &sourceView,
                               TargetType &targetView) {
  assert(Dim == sourceView.extent(1));
  using ExecSpace = typename SourceType::execution_space;
  const auto K = sourceView.extent(0);
  Kokkos::parallel_for(
      "MLS::FillPoints",
      Kokkos::RangePolicy<ExecSpace>(ExecSpace{}, 0, K),
      KOKKOS_LAMBDA(const std::size_t k) {
        ::ArborX::Point<Dim, coordinates_t> point{};
        for (int_t axis = 0; axis < Dim; ++axis) {
          point[axis] = sourceView(k, axis);
        }
        targetView(k) = point;
      });
}

/// @brief Builds a Moving Least Squares interpolator from a `Transfer` object
/// and writes the interpolated values into `rTransfer.targetValues`.
///
/// @tparam ExecSpace  Kokkos execution space.
/// @tparam Dim        Space dimension (1 <= Dim <= 3).
/// @param[in,out] rTransfer  Transfer object carrying source/target data.
PACMAN_MLS_FULL_TEMPLATE
PACMAN_MLS_CLASSNAME::MLSInterpolator(Transfer<ExecSpace, Dim> &rTransfer) {
  const std::string _region_name = "MLSInterpolator::MLSInterpolator";
  Kokkos::Profiling::ScopedRegion region(_region_name);

  const ExecSpace execspace{};
  using Point = ::ArborX::Point<Dim, coordinates_t>;
  using MemSpace = typename ExecSpace::memory_space;

  const int_t nTgt =
      static_cast<int_t>(rTransfer.targetPoints.extent(0));
  // Convert the 2-D target coordinate array into an ArborX::Point view.
  auto tgtPoints = Kokkos::View<Point *, MemSpace>(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "MLSInterpolator::tgtPoints"),
      nTgt);
  FillPoints<decltype(rTransfer.targetPoints), decltype(tgtPoints), Dim>(
      rTransfer.targetPoints, tgtPoints);
  Kokkos::fence();

  // Alias the output to the transfer's target-values view.
  this->out = rTransfer.targetValues;

  // Build Moving Least Squares coefficients and interpolate.
  ::ArborX::Interpolation::MovingLeastSquares<MemSpace> MLS(
      execspace, rTransfer.sourcePoints, tgtPoints);
  MLS.interpolate(execspace, rTransfer.sourceValues, this->out);
  Kokkos::fence();
}

} // namespace MLS
} // namespace PACMAN
