#ifndef UTILS_HPP
#define UTILS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
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

template <typename ExecSpace, int Dim, class Coordinates>
void print_clusters_view(
    Kokkos::View<ArborX::Point<Dim, Coordinates>**, ExecSpace>& clusters)
{
    auto clusters_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, clusters);
    for (size_t i = 0; i < clusters_h.extent(0); ++i)
    {
        std::cout << i << ". center: " << point_to_str(clusters_h(i, 0))
                  << std::endl;
        std::cout << i << ". points:" << std::endl;
        for (size_t j = 1; j < clusters_h.extent(1); ++j)
        {
            std::cout << "    - " << point_to_str(clusters_h(i, j))
                      << std::endl;
        }
    }
}

void print_cuda_memory_usage()
{
    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    std::cout << "used: " << (cuda_total - cuda_free) / 1000000 << "/"
              << (cuda_total) / 1000000 << "MB" << std::endl;
}

#endif /* ! UTILS_HPP */
