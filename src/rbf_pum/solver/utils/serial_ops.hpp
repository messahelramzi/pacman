#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "solver/utils/serial_solve.hpp"

namespace PACMAN {
namespace RbfPum {
template <KokkosViewRank<2> AViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType, RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void SerialFillA(const AViewType &A, const XType &X,
                                             const XOffsType &XOffs,
                                             const RbfFuncType &func) {
  const int_t N = A.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    const auto x = X(XOffs(i));
    for (int_t j = i; j < N; ++j) {
      const auto val = func.Eval(DistanceNoCheck(x, X(XOffs(j))));
      A(i, j) = val;
      A(j, i) = val;
    }
  }
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType, KokkosViewRank<1> YType,
          KokkosViewRank<1> YOffsType, RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
SerialFillE(AViewType &A, const XType &X, const XOffsType &XOffs,
            const YType &Y, const YOffsType &YOffs, const RbfFuncType &func) {
  const int_t N = A.extent_int(0);
  const int_t M = A.extent_int(1);
  for (int_t i = 0; i < N; ++i) {
    const auto y = Y(YOffs(i));
    for (int_t j = 0; j < M; ++j) {
      A(i, j) = func.Eval(DistanceNoCheck(y, X(XOffs(j))));
    }
  }
}

template <KokkosViewRank<1> BViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void SerialFillB(BViewType &B, const XType &X,
                                             const XOffsType &XOffs) {
  const int_t N = B.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    B(i) = X(XOffs(i));
  }
}

template <KokkosViewRank<1> SourceType, KokkosViewRank<1> OffsType,
          KokkosArray AxisType>
KOKKOS_INLINE_FUNCTION int_t SerialDeactivateOneAxis(const SourceType &src,
                                                     const OffsType &offs,
                                                     AxisType &activeAxis) {
  using ExecSpace = typename base_type<SourceType>::execution_space;
  constexpr int_t dim = activeAxis.size();
  Kokkos::Array<fp_t, dim> distances;
  for (int_t axis = 0; axis < dim; ++axis) {
    if (!activeAxis[axis]) {
      distances[axis] = fp_consts::max();
      continue;
    }
    auto min_max_pair = RbfPumMinMax(offs, [=](const int &i, const int &j) {
      return src(i)[axis] < src(j)[axis];
    });
    const fp_t dist = Kokkos::abs(src(min_max_pair.second)[axis] -
                                  src(min_max_pair.first)[axis]);
    distances[axis] = dist;
  }
  fp_t min = fp_consts::max();
  int_t min_i = 0;
  for (int_t axis = 0; axis < dim; ++axis) {
    if (distances[axis] < min) {
      min = distances[axis];
      min_i = axis;
    }
  }
  activeAxis[min_i] = false;
  return min_i;
}

template <KokkosViewRank<2> PViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> OffsViewType, KokkosArray AxisType>
KOKKOS_FUNCTION auto SerialFillPoly(PViewType &P, const XViewType &X,
                                    const OffsViewType &XOffs,
                                    AxisType &activeAxis) {
  constexpr int_t dim = activeAxis.size();
  const int_t N = P.extent_int(0);
  int_t active_count = 0;
  int_t active_index[dim];
  for (int_t j = 0; j < dim; ++j) {
    if (activeAxis[j]) {
      active_index[active_count++] = j;
    }
  }
  for (int_t i = 0; i < N; ++i) {
    P(i, 0) = fp_consts::one();
    const auto x = X(XOffs(i));
    for (int_t k = 0; k < active_count; ++k) {
      const int_t j = active_index[k];
      P(i, 1 + k) = x[j];
    }
  }
}

template <KokkosViewRank<2> QViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> XOffsType, KokkosViewRank<1> WViewType,
          KokkosArray AxisType>
KOKKOS_FORCEINLINE_FUNCTION int_t SerialFillQ(QViewType &Q, const XViewType &X,
                                              const XOffsType &XOffs,
                                              WViewType &W,
                                              AxisType &activeAxis) {
  using ExecSpace = typename QViewType::execution_space;
  constexpr int_t dim = activeAxis.size();
  int_t poly_vals = dim + 1;

  Kokkos::Array<fp_t, dim + 1> scratch_data;
  for (int i = 0; i < dim + 1; ++i)
  {
    scratch_data[i] = fp_consts::zero();
  }
  Kokkos::View<fp_t *, ExecSpace> scratch(scratch_data.data(), dim + 1);

  do {
    SerialFillPoly(Q, X, XOffs, activeAxis);
    auto S = Kokkos::subview(scratch, Kokkos::make_pair(0, poly_vals));
    SerialSVD(Q, S, W);
    const fp_t condition =
        S(0) / Kokkos::max(S(poly_vals - 1), fp_consts::epsilon());
    if (condition > 1.0e5) {
      SerialDeactivateOneAxis(X, XOffs, activeAxis);
      --poly_vals;
    } else {
      SerialFillPoly(Q, X, XOffs, activeAxis);
      break;
    }
  } while (poly_vals > 1);
  return poly_vals;
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> OutViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
SerialMulMatVec(const AViewType &A, const BViewType &B, OutViewType &out) {
  namespace KB = KokkosBlas;
  using Gemv =
      KB::SerialGemv<KB::Trans::NoTranspose, KB::Algo::Gemv::Unblocked>;
  Gemv::invoke(fp_consts::one(), A, B, fp_consts::zero(), out);
}

template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecAdd(UViewType &U,
                                                 const VViewType &V) {
  const int_t N = U.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    U(i) += V(i);
  }
}

template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecSub(UViewType &U,
                                                 const VViewType &V) {
  const int_t N = U.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    U(i) -= V(i);
  }
}
} // namespace RbfPum
} // namespace PACMAN
