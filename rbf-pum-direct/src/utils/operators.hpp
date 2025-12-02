#pragma once

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <iomanip>

namespace ArborX
{
    template <int Dim, class ScalarType>
    KOKKOS_INLINE_FUNCTION constexpr bool
    operator==(const ArborX::Point<Dim, ScalarType>& lhs,
               const ArborX::Point<Dim, ScalarType>& rhs)
    {
        for (int i = 0; i < Dim; ++i)
        {
            if (lhs[i] != rhs[i] && (lhs[i] == lhs[i] && rhs[i] == rhs[i]))
            {
                return false;
            }
        }
        return true;
    }

    template <int Dim, class ScalarType>
    KOKKOS_INLINE_FUNCTION constexpr bool
    operator!=(const ArborX::Point<Dim, ScalarType>& lhs,
               const ArborX::Point<Dim, ScalarType>& rhs)
    {
        return !(lhs == rhs);
    }

    template <int Dim, class ScalarType>
    KOKKOS_INLINE_FUNCTION bool constexpr
    operator<(const ArborX::Point<Dim, ScalarType>& lhs,
              const ArborX::Point<Dim, ScalarType>& rhs)
    {
        // Lexicographic ordering (NaNs are always bigger than non-NaNs)
        // follows strict weak ordering properties
        const bool is_lhs_nan = lhs[0] != lhs[0];
        const bool is_rhs_nan = rhs[0] != rhs[0];
        if (is_lhs_nan != is_rhs_nan)
        {
            return !is_lhs_nan;
        }
        for (int axis = 0; axis < Dim; ++axis)
        {
            if (lhs[axis] < rhs[axis])
            {
                return true;
            }
            if (lhs[axis] > rhs[axis])
            {
                return false;
            }
        }
        return false;
    }

    template <int Dim, class ScalarType>
    KOKKOS_INLINE_FUNCTION bool constexpr
    operator>(const ArborX::Point<Dim, ScalarType>& lhs,
              const ArborX::Point<Dim, ScalarType>& rhs)
    {
        return !(lhs == rhs) && !(lhs < rhs);
    }

    template <int Dim, class ScalarType>
    std::ostream& operator<<(std::ostream& os,
                             const ArborX::Point<Dim, ScalarType>& point)
    {
        std::ostringstream strs;
        strs << std::setprecision(16);
        strs << "ArborX::Point(";
        for (int i = 0; i < Dim - 1; i++)
        {
            strs << point[i] << ", ";
        }
        strs << point[Dim - 1] << ")";
        os << strs.str();
        return os;
    }
} // namespace ArborX

template <int Dim, class ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
squared_difference(const ArborX::Point<Dim, ScalarType>& lhs,
                   const ArborX::Point<Dim, ScalarType>& rhs)
{
    if (rhs[0] != rhs[0] || lhs[0] != lhs[0])
    {
        return static_cast<ScalarType>(-1);
    }
    ScalarType s = 0;
    for (int i = 0; i < Dim; ++i)
    {
        s += (rhs[i] - lhs[i]) * (rhs[i] - lhs[i]);
    }
    return s;
}

template <int Dim, class ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
NDdistance(const ArborX::Point<Dim, ScalarType>& lhs,
           const ArborX::Point<Dim, ScalarType>& rhs)
{
    const ScalarType d = squared_difference<Dim, ScalarType>(lhs, rhs);
    if (d < 0)
    {
        return static_cast<ScalarType>(-1);
    }
    return Kokkos::sqrt(d);
}

template <int Dim, class ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
squared_difference_no_check(const ArborX::Point<Dim, ScalarType>& lhs,
                            const ArborX::Point<Dim, ScalarType>& rhs)
{
    // clang-format off
    ScalarType acc = 0;
    for (int i = 0; i < Dim; ++i)
    {
        const ScalarType diff = rhs[i] - lhs[i];
        #ifdef FP_FAST_FMA
        acc = Kokkos::fma(diff, diff, acc);
        #else
        acc += diff * diff;
        #endif
    }
    // clang-format on
    return acc;
}

template <int Dim, class ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
NDdistance_no_check(const ArborX::Point<Dim, ScalarType>& lhs,
                    const ArborX::Point<Dim, ScalarType>& rhs)
{
    return Kokkos::sqrt(squared_difference_no_check<Dim, ScalarType>(lhs, rhs));
}

struct OffsetsScanPair
{
    int source_current;
    int target_current;
};

template <typename ExecSpace>
struct OffsetsScan
{
    Kokkos::View<int*, ExecSpace> source_offsets;
    Kokkos::View<int*, ExecSpace> target_offsets;

    Kokkos::View<int*, ExecSpace> rbf_mat_offsets;
    Kokkos::View<int*, ExecSpace> eval_mat_offsets;

    KOKKOS_FUNCTION void operator()(const int& i, OffsetsScanPair& pair,
                                    const bool& final) const
    {
        const int n = source_offsets(i + 1) - source_offsets(i);
        const int m = target_offsets(i + 1) - target_offsets(i);
        pair.source_current += n * n;
        pair.target_current += n * m;

        if (final)
        {
            rbf_mat_offsets(i + 1) = pair.source_current;
            eval_mat_offsets(i + 1) = pair.target_current;
        }
    }

    KOKKOS_FUNCTION void init(OffsetsScanPair& pair) const
    {
        pair.source_current = 0;
        pair.target_current = 0;
    }

    KOKKOS_FUNCTION void join(OffsetsScanPair& dest,
                              const OffsetsScanPair& src) const
    {
        dest.source_current += src.source_current;
        dest.target_current += src.target_current;
    }
};
