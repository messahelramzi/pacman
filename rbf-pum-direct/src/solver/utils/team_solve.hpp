#pragma once

#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_ApplyQ_TeamVector_Impl.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "concepts.hpp"

template <typename TeamHandle, KokkosViewRank<2> AViewType,
          KokkosViewRank<2> BViewType, KokkosViewRank<1> TViewType,
          KokkosViewRank<2> XViewType, KokkosViewRank<1> WViewType>
/**
 * @brief Solves AX=B for X using a QR factorization on A. Team version of
 * `SerialSolveQR`
 *
 * @param team_handle A `Kokkos::TeamHandleConcept` object which represent a
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
TeamSolveQR(const TeamHandle& team_handle, AViewType& A, const BViewType& B,
            TViewType& T, XViewType& X, WViewType& W)
{
    namespace KB = KokkosBatched;
    using QR = KB::TeamQR<TeamHandle, KB::Algo::QR::Unblocked>;
    using ApplyQ =
        KB::TeamVectorApplyQ<TeamHandle, KB::Side::Left, KB::Trans::Transpose,
                             KB::Algo::ApplyQ::Unblocked>;
    using CopyVector = KB::TeamCopy<TeamHandle, KB::Trans::NoTranspose, 1>;
    using Trsv =
        KB::TeamTrsv<TeamHandle, KB::Uplo::Upper, KB::Trans::NoTranspose,
                     KB::Diag::NonUnit, KB::Algo::Trsv::Unblocked>;

    // Computes A=QR (see dgeqp3(...) docs for details)
    QR::invoke(team_handle, A, T, W);
    // Copy B in X because we need B later
    CopyVector::invoke(team_handle, B, X);
    // Ax=B => QRx=B => (Q^T)QRx=(Q^T)B =>Rx=(Q^T)B
    // We store (Q^T)B in X
    ApplyQ::invoke(team_handle, A, T, X, W);
    // We better use subviews to solve the last system using Trsv (blas2)
    // instead of Trsm (blas 3)
    const int M = A.extent_int(1);
    auto x = Kokkos::subview(X, Kokkos::make_pair(0, M), 0);
    auto R =
        Kokkos::subview(A, Kokkos::make_pair(0, M), Kokkos::make_pair(0, M));
    // Solve Rx=(Q^T)B for x
    Trsv::invoke(team_handle, 1.0, R, x);
}

template <typename TeamHandle, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> BViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamSolveLU(const TeamHandle& team_handle,
                                             AViewType& A, BViewType& B)
{
    namespace KB = KokkosBatched;
    using LU = KB::TeamLU<TeamHandle, KB::Algo::LU::Unblocked>;
    using SolveLU = KB::TeamSolveLU<TeamHandle, KB::Trans::NoTranspose,
                                    KB::Algo::SolveLU::Unblocked>;

    LU::invoke(team_handle, A);
    team_handle.team_barrier();
    SolveLU::invoke(team_handle, A, B);
}
