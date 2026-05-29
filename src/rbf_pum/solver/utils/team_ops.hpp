//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"

namespace PACMAN {

namespace RbfPum {

/// @brief Fills the RBF matrix A in parallel. This function must not be called
/// in a nested parallel loop
/// @tparam TeamHandleType Type of the team of threads
/// @tparam AViewType Type of the matrix A (a concept enforces `Kokkos::View` of
/// rank 2)
/// @tparam XType Type of the source points view (a concept enforces
/// `Kokkos::View` of rank 1)
/// @tparam XOffsType Type of the access offsets associated to the source points
/// (a concept enforces `Kokkos::View` of rank 1)
/// @tparam RbfFuncType A RBF function (for instance: Wendland functions), a
/// concept enforces that function.Eval exists
/// @param teamHandle The Kokkos object which represents the team of threads
/// @param A The RBF matrix we are filling
/// @param X The source points contained in the local cluster
/// @param XOffs The access offsets of the local cluster points in the source
/// points view
/// @param func The RBF function we use to compute values to assign to A
/// @note A is symmetric, and is filled 2 values by 2 values using this property
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

/// @brief Fills the RBF evaluation matrix E in parallel. This function must not
/// be called in a nested parallel loop
/// @tparam TeamHandleType Type of the team of threads
/// @tparam AViewType Type of the matrix E (a concept enforces `Kokkos::View` of
/// rank 2)
/// @tparam XType Type of the source points view (a concept enforces
/// `Kokkos::View` of rank 1)
/// @tparam XOffsType Type of the access offsets associated to the source points
/// (a concept enforces `Kokkos::View` of rank 1)
/// @tparam YType Type of the target points view (a concept enforces
/// `Kokkos::View` of rank 1)
/// @tparam YOffsType Type of the access offsets associated to the target points
/// (a concept enforces `Kokkos::View` of rank 1)
/// @tparam RbfFuncType A RBF function (for instance: Wendland functions), a
/// concept enforces that function.Eval exists
/// @param teamHandle The Kokkos object which represents the team of threads
/// @param A The RBF evaluation matrix we are filling
/// @param X The source points contained in the local cluster
/// @param XOffs The access offsets of the local cluster points in the source
/// points view
/// @param Y The target points contained in the local cluster
/// @param YOffs The access offsets of the local cluster points in the target
/// points view
/// @param func The RBF function we use to compute values to assign to A
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

/// @brief Fills the vector of known source values B in parallel. This function
/// must not be called in a nested parallel loop
/// @tparam TeamHandleType Type of the team of threads
/// @tparam BViewType Type of the vector B (a concept enforces `Kokkos::View` of
/// rank 1)
/// @tparam XType Type of the source points view (a concept enforces
/// `Kokkos::View` of rank 1)
/// @tparam XOffsType Type of the access offsets associated to the source points
/// (a concept enforces `Kokkos::View` of rank 1)
/// @param teamHandle The Kokkos object which represents the team of threads
/// @param B The known source values vector B we are filling
/// @param X The source points contained in the local cluster
/// @param XOffs The access offsets of the local cluster points in the source
/// points view
template <typename TeamHandleType, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void TeamFillB(const TeamHandleType &teamHandle,
                                           BViewType &B, const XType &X,
                                           const XOffsType &XOffs) {
  const int_t N = B.extent_int(0);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(teamHandle, N),
                       [&](const int_t &i) { B(i) = X(XOffs(i)); });
}

} // namespace RbfPum

} // namespace PACMAN
