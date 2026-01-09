#pragma once

#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_ApplyQ_TeamVector_Impl.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"

namespace PACMAN {
namespace RbfPum {
template <typename TeamHandleType, KokkosViewRank<2> AViewType,
          KokkosViewRank<2> BViewType, KokkosViewRank<1> TViewType,
          KokkosViewRank<2> XViewType, KokkosViewRank<1> WViewType>
/**
 * @brief Solves AX=B for X using a QR factorization on A. Team version of
 * `SerialSolveQR`
 *
 * @param teamHandle A `Kokkos::TeamHandleConcept` object which represent a
 * team
 * @param A A rank 2 view of scalars, with the extents (N, M)
 * @param B A rank 2 view of scalars, of length N
 * @param T A rank 1 (non-initialized) view of scalars, of length M
 * @param X A rank 2 (non-initialized) view of scalars, of length N, it is the
 * output parameter
 * @param W A rank 1 (non-initialized) view of scalars, of length max(N, M), it
 * must be created with the LayoutRight argument
 * @return nothing
 * @note `B` is the only parameter which is guaranteed to not be modified
 * @note `T` and `W` can be allocated in the scratch memory space
 * @warning This function must not be called in an inner parallel loop
 */
KOKKOS_FORCEINLINE_FUNCTION auto
TeamSolveQR(const TeamHandleType &teamHandle, AViewType &A, const BViewType &B,
            TViewType &T, XViewType &X, WViewType &W) {
  namespace KB = KokkosBatched;
  using QR = KB::TeamQR<TeamHandleType, KB::Algo::QR::Unblocked>;
  using ApplyQ =
      KB::TeamVectorApplyQ<TeamHandleType, KB::Side::Left, KB::Trans::Transpose,
                           KB::Algo::ApplyQ::Unblocked>;
  using CopyVector = KB::TeamCopy<TeamHandleType, KB::Trans::NoTranspose, 1>;
  using Trsv =
      KB::TeamTrsv<TeamHandleType, KB::Uplo::Upper, KB::Trans::NoTranspose,
                   KB::Diag::NonUnit, KB::Algo::Trsv::Unblocked>;

  QR::invoke(teamHandle, A, T, W);
  CopyVector::invoke(teamHandle, B, X);
  ApplyQ::invoke(teamHandle, A, T, X, W);
  const int_t M = A.extent_int(1);
  auto x = Kokkos::subview(X, Kokkos::make_pair(0, M), 0);
  auto R = Kokkos::subview(A, Kokkos::make_pair(0, M), Kokkos::make_pair(0, M));
  Trsv::invoke(teamHandle, fp_consts::one(), R, x);
}

template <typename TeamHandleType, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> BViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamSolveLU(const TeamHandleType &teamHandle,
                                             AViewType &A, BViewType &B) {
  namespace KB = KokkosBatched;
  using LU = KB::TeamLU<TeamHandleType, KB::Algo::LU::Unblocked>;
  using SolveLU = KB::TeamSolveLU<TeamHandleType, KB::Trans::NoTranspose,
                                  KB::Algo::SolveLU::Unblocked>;

  LU::invoke(teamHandle, A);
  teamHandle.team_barrier();
  SolveLU::invoke(teamHandle, A, B);
}
} // namespace RbfPum
} // namespace PACMAN
