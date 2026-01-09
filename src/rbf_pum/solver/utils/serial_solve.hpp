#pragma once

#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"

namespace PACMAN {

namespace RbfPum {
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
 * must be created with the LayoutRight view argument
 * @return nothing
 * @note `B` is the only parameter which is guaranteed to not be modified. Caps
 * variables are matrices, others are vectors.
 */
KOKKOS_FORCEINLINE_FUNCTION auto SerialSolveQR(AViewType &A, const BViewType &B,
                                               TViewType &T, XViewType &X,
                                               WViewType &W) {
  namespace KB = KokkosBatched;
  using QR = KB::SerialQR<KB::Algo::QR::Unblocked>;
  using ApplyQ = KB::SerialApplyQ<KB::Side::Left, KB::Trans::Transpose,
                                  KB::Algo::ApplyQ::Unblocked>;
  using CopyVector = KB::SerialCopy<KB::Trans::NoTranspose, 2>;
  using Trsv = KB::SerialTrsv<KB::Uplo::Upper, KB::Trans::NoTranspose,
                              KB::Diag::NonUnit, KB::Algo::Trsv::Unblocked>;

  QR::invoke(A, T, W);
  CopyVector::invoke(B, X);
  ApplyQ::invoke(A, T, X, W);
  const int_t M = A.extent_int(1);
  auto x = Kokkos::subview(X, Kokkos::make_pair(0, M), 0);
  auto R = Kokkos::subview(A, Kokkos::make_pair(0, M), Kokkos::make_pair(0, M));
  Trsv::invoke(fp_consts::one(), R, x);
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> BViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialSolveLU(AViewType &A, BViewType &B) {
  namespace KB = KokkosBatched;
  using LU = KB::SerialLU<KB::Algo::LU::Unblocked>;
  using SolveLU =
      KB::SerialSolveLU<KB::Trans::NoTranspose, KB::Algo::SolveLU::Unblocked>;
  LU::invoke(A);
  SolveLU::invoke(A, B);
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> SViewType,
          KokkosViewRank<1> WViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialSVD(AViewType &A, SViewType &S,
                                           WViewType &W) {
  namespace KB = KokkosBatched;
  using SVD = KB::SerialSVD;

  constexpr fp_t tol = fp_consts::epsilon();
  constexpr int_t max_iters = 10000000;
  SVD::invoke(KB::SVD_S_Tag{}, A, S, W, tol, max_iters);
}

} // namespace RbfPum
} // namespace PACMAN
