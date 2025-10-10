#ifndef UTILS_HPP
#define UTILS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

template <int Dim, class Coordinates>
std::string point_to_str(const ArborX::Point<Dim, Coordinates>& point)
{
    std::ostringstream strs;
    strs << "ArborX::Point(";
    for (int i = 0; i < Dim - 1; i++)
    {
        strs << point[i] << ", ";
    }
    strs << point[Dim - 1] << ")";
    return strs.str();
}

template <int Dim, class Coordinates>
std::string sphere_to_str(const ArborX::Sphere<Dim, Coordinates>& sphere)
{
    std::ostringstream strs;
    strs << "ArborX::Sphere("
         << point_to_str<Dim, Coordinates>(sphere.centroid()) << ", "
         << sphere.radius() << ")";
    return strs.str();
}

template <int Dim, class Coordinates>
KOKKOS_INLINE_FUNCTION Coordinates
NDdistance(const ArborX::Point<Dim, Coordinates>& lhs,
           const ArborX::Point<Dim, Coordinates>& rhs)
{
    return std::sqrt(_NDdistance_without_sqrt<Dim, Coordinates>(lhs, rhs));
}

template <int Dim, class Coordinates>
KOKKOS_INLINE_FUNCTION size_t
_NDdistance_without_sqrt(const ArborX::Point<Dim, Coordinates>& lhs,
                         const ArborX::Point<Dim, Coordinates>& rhs)
{
    size_t s = 0;
    for (int i = 0; i < Dim; i++)
    {
        s += (rhs[i] - lhs[i]) * (rhs[i] - lhs[i]);
    }
    return s;
}

#endif /* ! UTILS_HPP */
