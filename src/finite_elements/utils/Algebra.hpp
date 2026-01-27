#pragma once

#include "common/types.hpp"

namespace PACMAN {
namespace FiniteElements {

template <typename ExecSpace, int_t Dim> struct Matrix;

template <typename ExecSpace> struct Matrix<ExecSpace, 1> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[1][1], ExecSpace> Amat) {
    Amat(0, 0) = 1.0 / Amat(0, 0);
  }
};

template <typename ExecSpace> struct Matrix<ExecSpace, 2> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[2][2], ExecSpace> Amat) {

    const fp_t a = Amat(0, 0), b = Amat(0, 1);
    const fp_t c = Amat(1, 0), d = Amat(1, 1);

    const fp_t det = a * d - b * c;

    const fp_t invDet = 1.0 / det;

    Amat(0, 0) = d * invDet;
    Amat(0, 1) = -b * invDet;
    Amat(1, 0) = -c * invDet;
    Amat(1, 1) = a * invDet;
  }
};

template <typename ExecSpace> struct Matrix<ExecSpace, 3> {
public:
  KOKKOS_INLINE_FUNCTION static void
  InvInPlace(Kokkos::View<fp_t[3][3], ExecSpace> Amat) {
    const fp_t a = Amat(0, 0), b = Amat(0, 1), c = Amat(0, 2);
    const fp_t d = Amat(1, 0), e = Amat(1, 1), f = Amat(1, 2);
    const fp_t g = Amat(2, 0), h = Amat(2, 1), i = Amat(2, 2);

    const fp_t det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

    const fp_t invDet = 1.0 / det;

    Amat(0, 0) = (e * i - f * h) * invDet;
    Amat(0, 1) = -(b * i - c * h) * invDet;
    Amat(0, 2) = (b * f - c * e) * invDet;

    Amat(1, 0) = -(d * i - f * g) * invDet;
    Amat(1, 1) = (a * i - c * g) * invDet;
    Amat(1, 2) = -(a * f - c * d) * invDet;

    Amat(2, 0) = (d * h - e * g) * invDet;
    Amat(2, 1) = -(a * h - b * g) * invDet;
    Amat(2, 2) = (a * e - b * d) * invDet;
  }
};
}; // namespace FiniteElements
} // namespace PACMAN
