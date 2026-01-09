#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"

namespace PACMAN {

namespace RbfPum {
template <typename TeamHandleType, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType,
          RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillA(const TeamHandleType &teamHandle, const AViewType &A, const XType &X,
          const XOffsType &XOffs, const RbfFuncType &func) {
  const int_t N = A.extent_int(0);
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(teamHandle, N), [&](const int_t &i) {
        const auto x = X(XOffs(i));
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(teamHandle, i, N), [&](const int &j) {
              const auto val = func.Eval(DistanceNoCheck(x, X(XOffs(j))));
              A(i, j) = val;
              A(j, i) = val;
            });
      });
}

template <typename TeamHandleType, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType,
          KokkosViewRank<1> YType, KokkosViewRank<1> YOffsType,
          RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillE(const TeamHandleType &teamHandle, AViewType &A, const XType &X,
          const XOffsType &XOffs, const YType &Y, const YOffsType &YOffs,
          const RbfFuncType &func) {
  const int_t N = A.extent_int(0);
  const int_t M = A.extent_int(1);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(teamHandle, N), [&](const int_t &i) {
        const auto y = Y(YOffs(i));
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(teamHandle, M), [&](const int_t &j) {
              A(i, j) = func.Eval(DistanceNoCheck(y, X(XOffs(j))));
            });
      });
}

template <typename TeamHandleType, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void TeamFillB(const TeamHandleType &teamHandle,
                                           BViewType &B, const XType &X,
                                           const XOffsType &XOffs) {
  const int_t N = B.extent_int(0);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(teamHandle, N),
                       [&](const int_t &i) { B(i) = X(XOffs(i)); });
}

template <typename TeamHandleType, KokkosViewRank<2> PViewType,
          KokkosViewRank<1> XViewType, KokkosViewRank<1> OffsViewType,
          KokkosArray AxisType>
KOKKOS_INLINE_FUNCTION auto
TeamPolyFill(const TeamHandleType &teamHandle, PViewType &P, const XViewType &X,
             const OffsViewType &XOffs, AxisType &activeAxis) {
  constexpr int_t dim = activeAxis.size();
  const int_t N = P.extent_int(0);
  int_t active_count = 0;
  int_t active_index[dim];
  for (int_t j = 0; j < dim; ++j) {
    if (activeAxis[j]) {
      active_index[active_count++] = j;
    }
  }

  Kokkos::parallel_for(Kokkos::TeamVectorMDRange(teamHandle, N, active_count),
                       [&](const int_t &i, const int_t &k) {
                         P(i, 1 + k) = X(XOffs(i))[active_index[k]];
                       });
}

template <typename TeamHandleType, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> BViewType, KokkosViewRank<1> OutViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
TeamMulMatVec(const TeamHandleType &teamHandle, const AViewType &A,
              const BViewType &B, OutViewType &out) {
  namespace KB = KokkosBlas;
  using Gemv = KB::TeamGemv<TeamHandleType, KB::Trans::NoTranspose,
                            KB::Algo::Gemv::Unblocked>;
  Gemv::invoke(teamHandle, fp_consts::one(), A, B, fp_consts::zero(), out);
}

template <typename TeamHandleType, KokkosViewRank<1> UViewType,
          KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecAdd(const TeamHandleType &teamHandle,
                                               UViewType &U,
                                               const VViewType &V) {
  const int_t N = U.extent_int(0);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(teamHandle, N),
                       [&](const int_t &i) { U(i) += V(i); });
}

template <typename TeamHandleType, KokkosViewRank<1> UViewType,
          KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecSub(const TeamHandleType &teamHandle,
                                               UViewType &U,
                                               const VViewType &V) {
  const int_t N = U.extent_int(0);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(teamHandle, N),
                       [&](const int_t &i) { U(i) -= V(i); });
}

} // namespace RbfPum

} // namespace PACMAN
