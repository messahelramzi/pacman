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
    auto d = squared_difference<Dim, ScalarType>(lhs, rhs);
    if (d < 0)
    {
        return static_cast<ScalarType>(-1);
    }
    return Kokkos::sqrt(d);
}