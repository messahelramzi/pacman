#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "concepts.hpp"
#include "solver/utils/serial_solve.hpp"

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType, RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void SerialFillA(const AViewType& A, const XType& X,
                                             const XOffsType& XOffs,
                                             const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    for (int i = 0; i < N; ++i)
    {
        const auto x = X(XOffs(i));
        for (int j = i; j < N; ++j)
        {
            const auto val = func.eval(NDdistance_no_check(x, X(XOffs(j))));
            A(i, j) = val;
            A(j, i) = val;
        }
    }
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType, KokkosViewRank<1> YType,
          KokkosViewRank<1> YOffsType, RBFFunction RbfFuncType>
KOKKOS_FORCEINLINE_FUNCTION void
SerialFillE(AViewType& A, const XType& X, const XOffsType& XOffs,
            const YType& Y, const YOffsType& YOffs, const RbfFuncType& func)
{
    const auto N = A.extent_int(0);
    const auto M = A.extent_int(1);
    for (int i = 0; i < N; ++i)
    {
        const auto y = Y(YOffs(i));
        for (int j = 0; j < M; ++j)
        {
            A(i, j) = func.eval(NDdistance_no_check(y, X(XOffs(j))));
        }
    }
}

template <KokkosViewRank<1> BViewType, KokkosViewRank<1> XType,
          KokkosViewRank<1> XOffsType>
KOKKOS_FORCEINLINE_FUNCTION void SerialFillB(BViewType& B, const XType& X,
                                             const XOffsType& XOffs)
{
    const auto N = B.extent_int(0);
    for (int i = 0; i < N; ++i)
    {
        B(i) = X(XOffs(i));
    }
}

template <KokkosViewRank<1> SourceType, KokkosViewRank<1> OffsType,
          KokkosArray AxisType>
KOKKOS_INLINE_FUNCTION int SerialDeactivateOneAxis(const SourceType& src,
                                                   const OffsType& offs,
                                                   AxisType& active_axis)
{
    using ExecSpace = typename base_type<SourceType>::execution_space;
    constexpr int m = active_axis.size();
    Kokkos::Array<double, m> distances;
    for (int axis = 0; axis < m; ++axis)
    {
        if (!active_axis[axis])
        {
            distances[axis] = KokkosKernels::ArithTraits<double>::max();
            continue;
        }
        auto span = MyMinMax(offs, [=](const int& a, const int& b) {
            return src(a)[axis] < src(b)[axis];
        });
        const auto dist =
            Kokkos::abs(src(span.second)[axis] - src(span.first)[axis]);
        distances[axis] = dist;
    }
    double min = KokkosKernels::ArithTraits<double>::max();
    int min_i = 0;
    for (int axis = 0; axis < m; ++axis)
    {
        if (distances[axis] < min)
        {
            min = distances[axis];
            min_i = axis;
        }
    }
    active_axis[min_i] = false;
    return min_i;
}

template <KokkosViewRank<2> PViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> OffsViewType, KokkosArray AxisType>
KOKKOS_FUNCTION auto SerialFillPoly(PViewType& P, const XViewType& X,
                                    const OffsViewType& XOffs,
                                    AxisType& active_axis)
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
    for (int i = 0; i < N; ++i)
    {
        P(i, 0) = 1.0;
        const auto x = X(XOffs(i));
        for (int k = 0; k < active_count; ++k)
        {
            const int j = active_index[k];
            P(i, 1 + k) = x[j];
        }
    }
}

template <KokkosViewRank<2> QViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> XOffsType, KokkosViewRank<1> WViewType,
          KokkosArray AxisType>
KOKKOS_FORCEINLINE_FUNCTION int SerialFillQ(QViewType& Q, const XViewType& X,
                                            const XOffsType& XOffs,
                                            WViewType& W, AxisType& active_axis)
{
    using ExecSpace = typename QViewType::execution_space;
    constexpr int dim = active_axis.size();
    int poly_vals = dim + 1;

    Kokkos::Array<double, dim + 1> scratch_data;
    Kokkos::View<double*, ExecSpace> scratch(scratch_data.data(), dim + 1);
    do
    {
        SerialFillPoly(Q, X, XOffs, active_axis);
        auto S = Kokkos::subview(scratch, Kokkos::make_pair(0, poly_vals));
        SerialSVD(Q, S, W);
        const double condition = S(0) / Kokkos::max(S(poly_vals - 1), 1.0e-14);
        if (condition > 1.0e5)
        {
            SerialDeactivateOneAxis(X, XOffs, active_axis);
            --poly_vals;
        }
        else
        {
            SerialFillPoly(Q, X, XOffs, active_axis);
            break;
        }
    } while (true);
    return poly_vals;
}

template <KokkosViewRank<2> AViewType, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> OutViewType>
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

template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecAdd(UViewType& U,
                                                 const VViewType& V)
{
    const int N = U.extent_int(0);
    for (int i = 0; i < N; ++i)
    {
        U(i) += V(i);
    }
}

template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecSub(UViewType& U,
                                                 const VViewType& V)
{
    const int N = U.extent_int(0);
    for (int i = 0; i < N; ++i)
    {
        U(i) -= V(i);
    }
}
