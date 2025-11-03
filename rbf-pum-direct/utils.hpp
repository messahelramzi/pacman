#ifndef UTILS_HPP
#define UTILS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#define DEBUG_INT(var)                                                         \
    std::cout << "[DEBUG] " << #var << " = " << (var) << std::endl;

#define DEBUG_FLOAT(var)                                                       \
    std::cout << "[DEBUG] " << #var << " = " << static_cast<double>(var)       \
              << std::endl;

#define DEBUG_STR(var)                                                         \
    std::cout << "[DEBUG] " << #var << " = \"" << (var) << "\"" << std::endl;

#define DEBUG_VIEW(view) print_view(view, "\n");

#define PRINT_STRING(string) std::cout << string << std::endl;

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
_NDdistance_without_sqrt(const ArborX::Point<Dim, Coordinates>& lhs,
                         const ArborX::Point<Dim, Coordinates>& rhs)
{
    Coordinates s = 0;
    for (int i = 0; i < Dim; i++)
    {
        if (rhs[i] != rhs[i] || lhs[i] != lhs[i])
        {
            return -1; // NaN
        }
        s += (rhs[i] - lhs[i]) * (rhs[i] - lhs[i]);
    }
    return s;
}

template <int Dim, class Coordinates>
KOKKOS_INLINE_FUNCTION Coordinates
NDdistance(const ArborX::Point<Dim, Coordinates>& lhs,
           const ArborX::Point<Dim, Coordinates>& rhs)
{
    auto d = _NDdistance_without_sqrt<Dim, Coordinates>(lhs, rhs);
    if (d < 0)
    {
        return -1;
    }
    return Kokkos::sqrt(d);
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
    std::cout << "nb_clusters: " << clusters_h.extent(0) << std::endl;
    std::cout << "nodes: " << clusters_h.extent(1) << std::endl;
}

void print_cuda_memory_usage()
{
    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    std::cout << "used: " << (cuda_total - cuda_free) / 1000000 << "/"
              << (cuda_total) / 1000000 << "MB" << std::endl;
}

template <typename ViewType>
void print_size_of_view(ViewType& v)
{
    size_t size = v.size() * sizeof(typename ViewType::value_type);
    std::cout << v.label() << " size: " << size << "b = " << size / 1'000'000.0
              << "Mb" << std::endl;
}

template <typename ViewType>
void print_view(ViewType& v, std::string sep = " ")
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        std::cout << m(i) << sep;
    }
    std::cout << std::endl;
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
}

template <typename ViewType>
void print_points_view(ViewType& v, std::string sep = " ")
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        std::cout << point_to_str(m(i)) << sep;
    }
    std::cout << std::endl;
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
}

template <typename ViewType>
void print_2d_points_view(ViewType& v, std::string sep = " ")
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        for (size_t j = 0; j < m.extent(1); ++j)
        {
            std::cout << point_to_str(m(i, j)) << sep;
        }
        std::cout << std::endl;
    }
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
    std::cout << v.label() << ".extent(1): " << m.extent(1) << std::endl;
}

template <typename ViewType>
void print_2d_view(ViewType& v, std::string sep = " ")
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        for (size_t j = 0; j < m.extent(1); ++j)
        {
            std::cout << std::setw(12) << m(i, j) << sep;
        }
        std::cout << std::endl;
    }
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
    std::cout << v.label() << ".extent(1): " << m.extent(1) << std::endl;
}

template <typename ViewType>
void print_pair_view(ViewType& v, std::string sep = " ")
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        std::cout << "<" << point_to_str(m(i).first) << ", "
                  << point_to_str(m(i).second) << ">" << sep;
    }
    std::cout << std::endl;
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
}

/* Operators overloads for ArborX::Point
 */
namespace ArborX
{
    template <int Dim, class Coordinates>
    KOKKOS_INLINE_FUNCTION constexpr bool
    operator==(const ArborX::Point<Dim, Coordinates>& lhs,
               const ArborX::Point<Dim, Coordinates>& rhs)
    {
        for (int i = 0; i < Dim; ++i)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }

    template <int Dim, class Coordinates>
    KOKKOS_INLINE_FUNCTION constexpr bool
    operator!=(const ArborX::Point<Dim, Coordinates>& lhs,
               const ArborX::Point<Dim, Coordinates>& rhs)
    {
        return !(lhs == rhs);
    }

    template <int Dim, class Coordinates>
    KOKKOS_INLINE_FUNCTION constexpr bool
    operator<(const ArborX::Point<Dim, Coordinates>& lhs,
              const ArborX::Point<Dim, Coordinates>& rhs)
    {
        return _NDdistance_without_sqrt<Dim, Coordinates>(
                   ArborX::Point<Dim, Coordinates>{}, lhs)
            < _NDdistance_without_sqrt<Dim, Coordinates>(
                   ArborX::Point<Dim, Coordinates>{}, rhs);
    }
} // namespace ArborX

template <typename ViewType>
std::string inline get_clusters_info(ViewType& nb_vertices_per_clusters)
{
    size_t maximum = 0;
    size_t minimum = std::numeric_limits<size_t>::max();
    double average = 0;
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      nb_vertices_per_clusters);
    for (size_t i = 0; i < view_h.extent(0); ++i)
    {
        if (view_h(i) > maximum)
        {
            maximum = view_h(i);
        }
        if (view_h(i) < minimum)
        {
            minimum = view_h(i);
        }
        average += view_h(i);
    }
    average /= (double)(view_h.extent(0));
    std::ostringstream strs;
    strs << "Number of clusters: " << nb_vertices_per_clusters.extent(0)
         << "\n";
    strs << "    Maximum: " << maximum << "\n";
    strs << "    Minimum: " << minimum << "\n";
    strs << "    Average: " << average;
    return strs.str();
}

#endif /* ! UTILS_HPP */
