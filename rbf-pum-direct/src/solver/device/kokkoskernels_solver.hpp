#pragma once
#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <Kokkos_Core.hpp>

template <typename TeamHandle, typename PsType, typename ValuesType,
          typename OffsType, typename PointsView>
KOKKOS_INLINE_FUNCTION auto
TeamFillPolynomial(const TeamHandle& team, PsType& Ps, const ValuesType& values,
                   const OffsType& offs, const PointsView& points,
                   const int stride)
{
    using data_type = typename PsType::non_const_value_type;

    const int k = team.league_rank();
    const int N = offs(k + 1) - offs(k);

    const auto range =
        Kokkos::make_pair(offs(k) * stride, offs(k + 1) * stride);
    auto P_data = Kokkos::subview(Ps, range);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](int i) {
        const int idx = values(offs(k) + i);
        const auto p = points(idx);

        P_data(i * stride) = data_type(1.0);

        for (int axis = 1; axis < stride; axis++)
        {
            P_data(i * stride + axis) = p[axis - 1];
        }
    });
}

template <typename TeamHandle, typename AsType, typename AsOffType,
          typename BsOffType>
KOKKOS_INLINE_FUNCTION auto
TeamGetSystemA(const TeamHandle& team_handle, const AsType& As,
               const AsOffType& As_offsets, const BsOffType& bs_offsets)
/**
 * @brief A_k = As(offsets(k):offsets(k + 1)).reshape(n,n)
 */
{
    using data_type = typename AsType::non_const_value_type;
    using exec_space = typename AsType::execution_space;
    const int k = team_handle.league_rank();
    const int n = bs_offsets(k + 1) - bs_offsets(k);
    const auto range = Kokkos::make_pair(As_offsets(k), As_offsets(k + 1));
    auto A_data = Kokkos::subview(As, range);

    Kokkos::View<data_type**, exec_space> A(A_data.data(), n, n);
    return A;
}

template <typename TeamHandle, typename BsType, typename BsOffType>
KOKKOS_INLINE_FUNCTION auto TeamGetSystemB(const TeamHandle& team_handle,
                                           const BsType& bs,
                                           const BsOffType& bs_offsets)
/**
 * @brief b_k = bs(offsets(k):offsets(k+1))
 */
{
    using data_type = typename BsType::non_const_value_type;
    using exec_space = typename BsType::execution_space;
    const int k = team_handle.league_rank();
    const int n = bs_offsets(k + 1) - bs_offsets(k);
    const auto range = Kokkos::make_pair(bs_offsets(k), bs_offsets(k + 1));
    auto b_data = Kokkos::subview(bs, range);

    Kokkos::View<data_type*, exec_space> b(b_data.data(), n);
    return b;
}

template <typename TeamHandle, typename WsType, typename WsOffType>
KOKKOS_INLINE_FUNCTION auto TeamGetSystemW(const TeamHandle& team_handle,
                                           const WsType& Ws,
                                           const WsOffType& ws_offsets)
{
    using data_type = typename WsType::non_const_value_type;
    using exec_space = typename WsType::execution_space;
    const int k = team_handle.league_rank();
    const int n = ws_offsets(k + 1) - ws_offsets(k);
    const auto range = Kokkos::make_pair(ws_offsets(k), ws_offsets(k + 1));
    auto w_data = Kokkos::subview(Ws, range);

    Kokkos::View<data_type*, Kokkos::LayoutRight, exec_space> w(w_data.data(),
                                                                n);
    return w;
}

template <typename TeamHandle, typename PsType, typename OffsType>
KOKKOS_INLINE_FUNCTION auto
TeamGetSystemP(const TeamHandle& team_handle, const PsType& Ps,
               const OffsType& offs, const int stride)
{
    using data_type = typename PsType::non_const_value_type;
    using exec_space = typename PsType::execution_space;
    const int k = team_handle.league_rank();
    const auto range =
        Kokkos::make_pair(offs(k) * stride, offs(k + 1) * stride);
    auto P_data = Kokkos::subview(Ps, range);

    Kokkos::View<data_type**, exec_space> P(P_data.data(),
                                            offs(k + 1) - offs(k), stride);
    return P;
}

template <typename TeamHandle, typename AType, typename BType, typename XType,
          typename TType, typename WType>
KOKKOS_INLINE_FUNCTION auto TeamSolveQR(const TeamHandle& team_handle, AType& A,
                                        BType& b, XType& x, TType& t, WType& w)
{
    KokkosBatched::TeamVectorQR<
        TeamHandle, KokkosBatched::Algo::QR::Unblocked>::invoke(team_handle, A,
                                                                t, w);
    team_handle.team_barrier();

    KokkosBatched::TeamVectorCopy<TeamHandle, KokkosBatched::Trans::NoTranspose,
                                  1>::invoke(team_handle, b, x);
    team_handle.team_barrier();

    auto sol = Kokkos::subview(x, Kokkos::make_pair(0, A.extent_int(1)));

    KokkosBatched::TeamVectorApplyQ<
        TeamHandle, KokkosBatched::Side::Left,
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::ApplyQ::Unblocked>::invoke(team_handle, A, t, sol,
                                                        w);
    team_handle.team_barrier();
}

template <typename TeamHandle, typename AType, typename BType>
KOKKOS_INLINE_FUNCTION void TeamSolveLU(const TeamHandle& team_handle, AType& A,
                                        BType& b)
{
    KokkosBatched::TeamLU<
        TeamHandle, KokkosBatched::Algo::LU::Unblocked>::invoke(team_handle, A);
    team_handle.team_barrier();

    KokkosBatched::TeamSolveLU<
        TeamHandle, KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::SolveLU::Unblocked>::invoke(team_handle, A, b);
}
