//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <iomanip>

#include "common/types.hpp"
#include "utils/utils.hpp"

namespace ArborX {

/// @brief Custom operator overload for `operator==` to compare two
/// `ArborX::Point` using the values of each point, and handle NaN by skipping
/// them
/// @tparam Dim The space dimension of the points
/// @param lhs The `lhs` point in `lhs == rhs`
/// @param rhs The `rhs` point in `lhs == rhs`
template <int Dim>
KOKKOS_INLINE_FUNCTION constexpr bool
operator==(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
           const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  for (int i = 0; i < Dim; ++i) {
    if (lhs[i] != rhs[i] && (lhs[i] == lhs[i] && rhs[i] == rhs[i])) {
      return false;
    }
  }
  return true;
}

/// @brief Custom operator overload for `operator!=` to compare two
/// `ArborX::Point` using the values of each point, and handle NaN by skipping
/// them
/// @tparam Dim The space dimension of the points
/// @param lhs The `lhs` point in `lhs != rhs`
/// @param rhs The `rhs` point in `lhs != rhs`
/// @returns `!(lhs == rhs)`
template <int Dim>
KOKKOS_INLINE_FUNCTION constexpr bool
operator!=(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
           const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  return !(lhs == rhs);
}

/// @brief Custom operator overload for `operator<` to compare two
/// `ArborX::Point` using the values of each point. Performs a lexicographic
/// ordering (NaNs are always bigger than non-NaNs)
///        and follows strict weak ordering properties
/// @tparam Dim The space dimension of the points
/// @param lhs The `lhs` point in `lhs < rhs`
/// @param rhs The `rhs` point in `lhs < rhs`
/// @returns `true` if lhs < rhs
template <int Dim>
KOKKOS_INLINE_FUNCTION bool constexpr
operator<(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
          const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  const bool is_lhs_nan = lhs[0] != lhs[0];
  const bool is_rhs_nan = rhs[0] != rhs[0];
  if (is_lhs_nan != is_rhs_nan) {
    return !is_lhs_nan;
  }
  for (int axis = 0; axis < Dim; ++axis) {
    if (lhs[axis] < rhs[axis]) {
      return true;
    }
    if (lhs[axis] > rhs[axis]) {
      return false;
    }
  }
  return false;
}

/// @brief Custom operator overload for `operator>` to compare two
/// `ArborX::Point` using the values of each point. Performs a lexicographic
/// ordering (NaNs are always bigger than non-NaNs)
///        and follows strict weak ordering properties
/// @tparam Dim The space dimension of the points
/// @param lhs The `lhs` point in `lhs > rhs`
/// @param rhs The `rhs` point in `lhs > rhs`
/// @returns !(lhs <= rhs)
template <int Dim>
KOKKOS_INLINE_FUNCTION bool constexpr
operator>(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
          const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  return !(lhs == rhs) && !(lhs < rhs);
}

/// @brief Custom operator overload for `operator<<` to print a `ArborX::Point`
/// with a nice format (ArborX::Point(x, y, z)). The coordinates are formated
/// using their type
/// @warning Works on host backends only
/// @tparam Dim The space dimension of the points
/// @param point The point to print
/// @returns The stream the point has been added to
template <int Dim>
std::ostream &
operator<<(std::ostream &os,
           const ArborX::Point<Dim, PACMAN::coordinates_t> &point) {
  std::ostringstream strs;
  strs << PACMAN::fp_consts::set_precision();
  strs << "ArborX::Point(";
  for (int i = 0; i < Dim - 1; i++) {
    strs << point[i] << ", ";
  }
  strs << point[Dim - 1] << ")";
  os << strs.str();
  return os;
}
} // namespace ArborX

namespace PACMAN {
namespace RbfPum {

/// @brief Returns the squared distance between two points, and check for NaN
/// values
/// @tparam Dim The space dimension of both points
/// @param lhs Source `ArborX::Point`
/// @param rhs Target `ArborX::Point`
/// @return $\sigma_{i=0}^{Dim}\left(rhs_i - lhs_i)^2\right)$ if both `lhs` and
/// `rhs` are valid (no NaN value) else returns -1.0
template <int_t Dim>
KOKKOS_INLINE_FUNCTION fp_t
SquaredDifference(const ::ArborX::Point<Dim, coordinates_t> &lhs,
                  const ::ArborX::Point<Dim, coordinates_t> &rhs) noexcept {
  if (rhs[0] != rhs[0] || lhs[0] != lhs[0]) {
    return -fp_consts::one();
  }
  fp_t acc = fp_consts::zero();
  for (int_t i = 0; i < Dim; ++i) {
    acc += (rhs[i] - lhs[i]) * (rhs[i] - lhs[i]);
  }
  return acc;
}

/// @brief Returns the euclidian norm between two points by performing a square
/// root operation on the `SquaredDifference` result
/// @tparam Dim The space dimension of both points
/// @param lhs Source `ArborX::Point`
/// @param rhs Target `ArborX::Point`
/// @note see `SquaredDifference`, if `SquaredDifference` returns -1.0, this
/// function returns -1.0 as well
template <int_t Dim>
KOKKOS_INLINE_FUNCTION fp_t
Distance(const ::ArborX::Point<Dim, coordinates_t> &lhs,
         const ::ArborX::Point<Dim, coordinates_t> &rhs) {
  const fp_t d = SquaredDifference(lhs, rhs);
  if (d < fp_consts::zero()) {
    return -fp_consts::one();
  }
  return Kokkos::sqrt(d);
}

/// @brief Returns the squared distance between two points, and don't check for
/// NaN values
/// @tparam Dim The space dimension of both points
/// @param lhs Source `ArborX::Point`
/// @param rhs Target `ArborX::Point`
/// @return $\sigma_{i=0}^{Dim}\left(rhs_i - lhs_i)^2\right)$
template <int_t Dim>
KOKKOS_INLINE_FUNCTION fp_t
SquaredDifferenceNoCheck(const ::ArborX::Point<Dim, coordinates_t> &lhs,
                         const ::ArborX::Point<Dim, coordinates_t> &rhs) {
  fp_t acc = fp_consts::zero();
  for (int_t i = 0; i < Dim; ++i) {
    acc += (rhs[i] - lhs[i]) * (rhs[i] - lhs[i]);
  }
  return acc;
}

/// @brief Returns the euclidian norm between two points by performing a square
/// root operation on the `SquaredDifferenceNoCheck` result
/// @tparam Dim The space dimension of both points
/// @param lhs Source `ArborX::Point`
/// @param rhs Target `ArborX::Point`
/// @note see `SquaredDifferenceNoCheck`
template <int_t Dim>
KOKKOS_INLINE_FUNCTION fp_t
DistanceNoCheck(const ::ArborX::Point<Dim, coordinates_t> &lhs,
                const ::ArborX::Point<Dim, coordinates_t> &rhs) {
  return Kokkos::sqrt(SquaredDifferenceNoCheck(lhs, rhs));
}

struct OffsetsScanPair {
  offset_t sourceCurrent;
  offset_t targetCurrent;
};

/// @brief Struct used to build multiple matrices access offsets with one scan
/// loop only, during the systems solver step
/// @tparam ExecSpace The Kokkos execution space the operation happens in
template <typename ExecSpace> struct OffsetsScan {
  Kokkos::View<offset_t *, ExecSpace> sourceOffsets;
  Kokkos::View<offset_t *, ExecSpace> targetOffsets;

  Kokkos::View<offset_t *, ExecSpace> rbfMatOffsets;
  Kokkos::View<offset_t *, ExecSpace> evalMatOffsets;

  KOKKOS_FUNCTION void operator()(const int &i, OffsetsScanPair &pair,
                                  const bool &final) const {
    const offset_t n = sourceOffsets(i + 1) - sourceOffsets(i);
    const offset_t m = targetOffsets(i + 1) - targetOffsets(i);
    pair.sourceCurrent += n * n;
    pair.targetCurrent += n * m;

    if (final) {
      rbfMatOffsets(i + 1) = pair.sourceCurrent;
      evalMatOffsets(i + 1) = pair.targetCurrent;
    }
  }

  KOKKOS_FUNCTION void init(OffsetsScanPair &pair) const {
    pair.sourceCurrent = 0;
    pair.targetCurrent = 0;
  }

  KOKKOS_FUNCTION void join(OffsetsScanPair &dest,
                            const OffsetsScanPair &src) const {
    dest.sourceCurrent += src.sourceCurrent;
    dest.targetCurrent += src.targetCurrent;
  }
};
} // namespace RbfPum
} // namespace PACMAN
