#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_team_gemv.hpp>
#include <Kokkos_Core.hpp>

template <typename AsViewType, typename OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfMatrix(const int& k, const AsViewType& As, const OffsViewType& offs,
             const int& n, const int& m)
{
    using data_type = typename AsViewType::non_const_value_type;
    using memory_space = typename AsViewType::memory_space;
    const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
    auto A_data = Kokkos::subview(As, slice);
    if (A_data.extent_int(0) != n * m)
    {
        Kokkos::abort("GetRbfMatrix: Invalid subview size!\n");
    }
    return Kokkos::View<data_type**, Kokkos::LayoutLeft, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(A_data.data(),
                                                                 n, m);
}

template <typename PsViewType, typename OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetPolyMatrix(const int& k, const PsViewType& Ps, const OffsViewType& offs,
              const int& n, const int& m)
{
    using data_type = typename PsViewType::non_const_value_type;
    using memory_space = typename PsViewType::memory_space;
    const auto slice = Kokkos::make_pair(offs(k) * m, offs(k + 1) * m);
    auto P_data = Kokkos::subview(Ps, slice);
    if (P_data.extent_int(0) != n * m)
    {
        Kokkos::abort("GetPolyMatrix: Invalid subview size!\n");
    }
    return Kokkos::View<data_type**, Kokkos::LayoutLeft, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(P_data.data(),
                                                                 n, m);
}

template <typename BsViewType, typename OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfVector(const int& k, const BsViewType& Bs, const OffsViewType& offs,
             const int& n)
{
    using data_type = typename BsViewType::non_const_value_type;
    using memory_space = typename BsViewType::memory_space;
    const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
    auto B_data = Kokkos::subview(Bs, slice);
    if (B_data.extent_int(0) != n)
    {
        Kokkos::abort("GetRbfVector: Invalid subview size!\n");
    }
    return Kokkos::View<data_type*, Kokkos::LayoutLeft, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(B_data.data(),
                                                                 n);
}

template <typename TeamHandle, typename AViewType, typename XType,
          typename XOffsType, typename RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillMatrixA(const TeamHandle& team_handle, const AViewType& A,
                const XType& X, const XOffsType& XOffs, const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    const auto M = A.extent_int(1);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), KOKKOS_LAMBDA(const int& i) {
            const auto source_point = X(XOffs(i));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_handle, i, M),
                                 [&](const int& j) {
                                     const auto val =
                                         func.eval(NDdistance_no_check(
                                             source_point, X(XOffs(j))));
                                     A(i, j) = val;
                                     A(j, i) = val;
                                 });
        });
}

template <typename PViewType, typename XType, typename XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void SerialFillPoly(PViewType& P, const XType& X,
                                                const XOffsType& XOffs)
{
    using exec_space = typename PViewType::execution_space;
    const auto N = P.extent_int(0);
    const auto M = P.extent_int(1);
    for (int i = 0; i < N; ++i)
    {
        const auto source_point = X(XOffs(i));
        for (int j = 0; j < M; ++j)
        {
            char mask = static_cast<char>(j != 0);
            P(i, j) = !mask * 1.0 + mask * source_point[j - mask];
        }
    }
}

template <typename TeamHandle, typename PViewType, typename XType,
          typename XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void TeamFillPoly(const TeamHandle& team_handle,
                                              PViewType& P, const XType& X,
                                              const XOffsType& XOffs)
{
    const auto N = P.extent_int(0);
    const auto M = P.extent_int(1);
    Kokkos::parallel_for(
        Kokkos::TeamVectorMDRange(team_handle, N, M),
        KOKKOS_LAMBDA(const int& i, const int& j) {
            char mask = static_cast<char>(j != 0);
            P(i, j) = !mask * 1.0 + mask * X(XOffs(i))[j - mask];
        });
}

template <typename TeamHandle, typename VViewType, typename XType,
          typename XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void TeamFillVec(const TeamHandle& team_handle,
                                             VViewType& V, const XType& X,
                                             const XOffsType& XOffs)
{
    const auto N = V.extent_int(0);
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team_handle, N),
        KOKKOS_LAMBDA(const int& i) { V(i) = X(XOffs(i)); });
}

template <typename TeamHandle, typename AViewType, typename XType,
          typename XOffsType, typename YType, typename YOffsType,
          typename RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamFillEvalMat(const TeamHandle& team_handle, AViewType& A, const XType& X,
                const XOffsType& XOffs, const YType& Y, const YOffsType& YOffs,
                const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    const auto M = A.extent_int(1);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), KOKKOS_LAMBDA(const int& i) {
            const auto target_point = Y(YOffs(i));
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_handle, M), [&](const int& j) {
                    A(i, j) = func.eval(
                        NDdistance_no_check(target_point, X(XOffs(j))));
                });
        });
}

template <typename AViewType, typename BViewType, typename OutViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
SerialMulMatVec(const AViewType& A, const BViewType& B, OutViewType& out)
{
    namespace KB = KokkosBlas;
    using Gemv =
        KB::SerialGemv<KB::Trans::NoTranspose, KB::Algo::Gemv::Unblocked>;
    using data_type = typename AViewType::non_const_value_type;
    constexpr data_type zero = static_cast<data_type>(0.0);
    constexpr data_type one = static_cast<data_type>(1.0);
    Gemv::invoke(one, A, B, zero, out);
}

template <typename TeamHandle, typename AViewType, typename BViewType,
          typename OutViewType>
KOKKOS_FORCEINLINE_FUNCTION void
TeamMulMatVec(const TeamHandle& team_handle, const AViewType& A,
              const BViewType& B, OutViewType& out)
{
    namespace KB = KokkosBlas;
    using Gemv = KB::TeamVectorGemv<TeamHandle, KB::Trans::NoTranspose,
                                    KB::Algo::Gemv::Unblocked>;
    using data_type = typename AViewType::non_const_value_type;
    constexpr data_type zero = static_cast<data_type>(0.0);
    constexpr data_type one = static_cast<data_type>(1.0);
    Gemv::invoke(team_handle, one, A, B, zero, out);
}

template <typename UViewType, typename VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecSub(UViewType& U,
                                                 const VViewType& V)
{
    const int N = U.extent_int(0);
    for (int i = 0; i < N; ++i)
    {
        U(i) -= V(i);
    }
}

template <typename TeamHandle, typename UViewType, typename VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecSub(const TeamHandle& team_handle,
                                               UViewType& U, const VViewType& V)
{
    const int N = U.extent_int(0);
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team_handle, N),
        KOKKOS_LAMBDA(const int& i) { U(i) -= V(i); });
}

template <typename TeamHandle, typename UViewType, typename VViewType>
KOKKOS_FORCEINLINE_FUNCTION void TeamVecVecAdd(const TeamHandle& team_handle,
                                               UViewType& U, const VViewType& V)
{
    const int N = U.extent_int(0);
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team_handle, N),
        KOKKOS_LAMBDA(const int& i) { U(i) += V(i); });
}
