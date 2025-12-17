#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "concepts.hpp"

template <typename TeamHandle, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType,
          RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillA(const TeamHandle& team_handle, const AViewType& A, const XType& X,
          const XOffsType& XOffs, const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), [&](const int& i) {
            const auto x = X(XOffs(i));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_handle, i, N),
                                 [&](const int& j) {
                                     const auto val = func.eval(
                                         NDdistance_no_check(x, X(XOffs(j))));
                                     A(i, j) = val;
                                     A(j, i) = val;
                                 });
        });
}

template <typename TeamHandle, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType,
          KokkosViewRank<1> YType, KokkosViewRank<1> YOffsType,
          RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillE(const TeamHandle& team_handle, AViewType& A, const XType& X,
          const XOffsType& XOffs, const YType& Y, const YOffsType& YOffs,
          const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    const auto M = A.extent_int(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), [&](const int& i) {
            const auto y = Y(YOffs(i));
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_handle, M), [&](const int& j) {
                    A(i, j) = func.eval(NDdistance_no_check(y, X(XOffs(j))));
                });
        });
}

template <typename TeamHandle, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> XType, KokkosViewRank<1> XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void TeamFillB(const TeamHandle& team_handle,
                                           BViewType& B, const XType& X,
                                           const XOffsType& XOffs)
{
    const auto N = B.extent_int(0);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_handle, N),
                         [&](const int& i) { B(i) = X(XOffs(i)); });
}

template <typename TeamHandle, KokkosViewRank<2> PViewType,
          KokkosViewRank<1> XViewType, KokkosViewRank<1> OffsViewType,
          KokkosArray AxisType>
KOKKOS_INLINE_FUNCTION auto
TeamPolyFill(const TeamHandle& team_handle, PViewType& P, const XViewType& X,
             const OffsViewType& XOffs, AxisType& active_axis)
{
    constexpr int dim = active_axis.size();
    const int N = P.extent_int(0);
    int active_count = 0;
    int active_index[dim];
    for (int j = 0; j < dim; ++j)
    {
        if (active_axis[j])
        {
            active_index[active_count++] = j;
        }
    }

    Kokkos::parallel_for(
        Kokkos::TeamVectorMDRange(team_handle, N, active_count),
        [&](const int& i, const int& k) {
            P(i, 1 + k) = X(XOffs(i))[active_index[k]];
        });
}

template <typename TeamHandle, KokkosViewRank<2> AViewType,
          KokkosViewRank<1> BViewType, KokkosViewRank<1> OutViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
TeamMulMatVec(const TeamHandle& team_handle, const AViewType& A,
              const BViewType& B, OutViewType& out)
{
    namespace KB = KokkosBlas;
    using Gemv = KB::TeamGemv<TeamHandle, KB::Trans::NoTranspose,
                              KB::Algo::Gemv::Unblocked>;
    using data_type = typename AViewType::non_const_value_type;
    constexpr data_type zero = static_cast<data_type>(0.0);
    constexpr data_type one = static_cast<data_type>(1.0);
    Gemv::invoke(team_handle, one, A, B, zero, out);
}

template <typename TeamHandle, KokkosViewRank<1> UViewType,
          KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecAdd(const TeamHandle& team_handle,
                                               UViewType& U, const VViewType& V)
{
    const int N = U.extent_int(0);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_handle, N),
                         [&](const int& i) { U(i) += V(i); });
}

template <typename TeamHandle, KokkosViewRank<1> UViewType,
          KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecSub(const TeamHandle& team_handle,
                                               UViewType& U, const VViewType& V)
{
    const int N = U.extent_int(0);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_handle, N),
                         [&](const int& i) { U(i) -= V(i); });
}
