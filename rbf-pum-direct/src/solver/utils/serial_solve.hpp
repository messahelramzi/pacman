#pragma once

#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "concepts.hpp"

template <KokkosViewRank<2> AViewType, KokkosViewRank<2> BViewType,
          KokkosViewRank<1> TViewType, KokkosViewRank<2> XViewType,
          KokkosViewRank<1> WViewType>
/**
 * @brief Solves AX=B for X using a QR factorization on A
 *
 * @param A A rank 2 view of scalars, with the extents (N, M)
 * @param B A rank 2 view of scalars, with the extents (N, 1)
 * @param T A rank 1 (non-initialized) view of scalars, of length M
 * @param X A rank 2 (non-initialized) view of scalars, of length (N, 1), it is
 * the output parameter (only the M first values are relevant as an output)
 * @param W A rank 1 (non-initialized) view of scalars, of length max(N, M), it
 * must be created with the LayoutRight argument
 * @return nothing
 * @note `B` is the only parameter which is guaranteed to not be modified
 */
KOKKOS_FORCEINLINE_FUNCTION auto SerialSolveQR(AViewType& A, const BViewType& B,
                                               TViewType& T, XViewType& X,
                                               WViewType& W)
{
    namespace KB = KokkosBatched;
    using QR = KB::SerialQR<KB::Algo::QR::Unblocked>;
    using ApplyQ = KB::SerialApplyQ<KB::Side::Left, KB::Trans::Transpose,
                                    KB::Algo::ApplyQ::Unblocked>;
    using CopyVector = KB::SerialCopy<KB::Trans::NoTranspose, 2>;
    using Trsv = KB::SerialTrsv<KB::Uplo::Upper, KB::Trans::NoTranspose,
                                KB::Diag::NonUnit, KB::Algo::Trsv::Unblocked>;

    // Computes A=QR (see dgeqp3(...) docs for details)
    QR::invoke(A, T, W);
    // Copy B in X because we need B later
    CopyVector::invoke(B, X);
    // Ax=B => QRx=B => (Q^T)QRx=(Q^T)B =>Rx=(Q^T)B
    // We store (Q^T)B in X
    ApplyQ::invoke(A, T, X, W);
    // We better use subviews to solve the last system using Trsv (blas2)
    // instead of Trsm (blas 3)
    const int M = A.extent_int(1);
    auto x = Kokkos::subview(X, Kokkos::make_pair(0, M), 0);
    auto R =
        Kokkos::subview(A, Kokkos::make_pair(0, M), Kokkos::make_pair(0, M));
    // Solve Rx=(Q^T)B for x
    Trsv::invoke(1.0, R, x);
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> BViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialSolveLU(AViewType& A, BViewType& B)
{
    namespace KB = KokkosBatched;
    using LU = KB::SerialLU<KB::Algo::LU::Unblocked>;
    using SolveLU =
        KB::SerialSolveLU<KB::Trans::NoTranspose, KB::Algo::SolveLU::Unblocked>;

    LU::invoke(A);
    SolveLU::invoke(A, B);
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> SViewType,
          KokkosViewRank<1> WViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialSVD(AViewType& A, SViewType& S,
                                           WViewType& W)
{
    namespace KB = KokkosBatched;
    using SVD = KB::SerialSVD;

    constexpr double tol = 10e-14;
    constexpr int max_iters = 10000000;
    SVD::invoke(KB::SVD_S_Tag{}, A, S, W, tol, max_iters);
}
