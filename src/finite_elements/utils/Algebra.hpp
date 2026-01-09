#pragma once

#include "common/types.hpp"

namespace PACMAN {
namespace FiniteElements {
// TODO: fix aliasing?
template <typename ExecSpace, int spaceDimension> struct Matrix;

template <typename ExecSpace> struct Matrix<ExecSpace, 1> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[1][1], ExecSpace> Amat) {
    Amat(0, 0) = 1.0 / Amat(0, 0);
    return;
  }
};

template <typename ExecSpace> struct Matrix<ExecSpace, 2> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[2][2], ExecSpace> Amat) {
    const fp_t one_over_det =
        1.0 / (Amat(0, 0) * Amat(1, 1) - Amat(1, 0) * Amat(0, 1));

    Amat(0, 0) = Amat(1, 1) * one_over_det;
    Amat(1, 0) = -Amat(1, 0) * one_over_det;
    Amat(0, 1) = -Amat(0, 1) * one_over_det;
    Amat(1, 1) = Amat(0, 0) * one_over_det;

    return;
  }
};

template <typename ExecSpace> struct Matrix<ExecSpace, 3> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[3][3], ExecSpace> Amat) {
    const fp_t one_over_det =
        1.0 /
        (Amat(0, 0) * (Amat(1, 1) * Amat(2, 2) - Amat(1, 2) * Amat(2, 1)) -
         Amat(0, 1) * (Amat(1, 0) * Amat(2, 2) - Amat(1, 2) * Amat(2, 0)) +
         Amat(0, 2) * (Amat(1, 0) * Amat(2, 1) - Amat(1, 1) * Amat(2, 0)));

    Amat(0, 0) =
        (Amat(1, 1) * Amat(2, 2) - Amat(2, 1) * Amat(1, 2)) * one_over_det;
    Amat(0, 1) =
        (Amat(0, 2) * Amat(2, 1) - Amat(0, 1) * Amat(2, 2)) * one_over_det;
    Amat(0, 2) =
        (Amat(0, 1) * Amat(1, 2) - Amat(0, 2) * Amat(1, 1)) * one_over_det;

    Amat(1, 0) =
        (Amat(1, 2) * Amat(2, 0) - Amat(1, 0) * Amat(2, 2)) * one_over_det;
    Amat(1, 1) =
        (Amat(0, 0) * Amat(2, 2) - Amat(0, 2) * Amat(2, 0)) * one_over_det;
    Amat(1, 2) =
        (Amat(1, 0) * Amat(0, 2) - Amat(0, 0) * Amat(1, 2)) * one_over_det;

    Amat(2, 0) =
        (Amat(1, 0) * Amat(2, 1) - Amat(2, 0) * Amat(1, 1)) * one_over_det;
    Amat(2, 1) =
        (Amat(2, 0) * Amat(0, 1) - Amat(0, 0) * Amat(2, 1)) * one_over_det;
    Amat(2, 2) =
        (Amat(0, 0) * Amat(1, 1) - Amat(1, 0) * Amat(0, 1)) * one_over_det;

    return;
  }
};

}; // namespace FiniteElements

} // namespace PACMAN
