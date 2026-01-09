#pragma once

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <iomanip>

#include "common/types.hpp"
#include "utils/utils.hpp"

namespace ArborX {
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

template <int Dim>
KOKKOS_INLINE_FUNCTION constexpr bool
operator!=(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
           const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  return !(lhs == rhs);
}

template <int Dim>
KOKKOS_INLINE_FUNCTION bool constexpr
operator<(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
          const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  // Lexicographic ordering (NaNs are always bigger than non-NaNs)
  // follows strict weak ordering properties
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

template <int Dim>
KOKKOS_INLINE_FUNCTION bool constexpr
operator>(const ArborX::Point<Dim, PACMAN::coordinates_t> &lhs,
          const ArborX::Point<Dim, PACMAN::coordinates_t> &rhs) noexcept {
  return !(lhs == rhs) && !(lhs < rhs);
}

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
