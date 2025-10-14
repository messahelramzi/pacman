#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <ArborX_Box.hpp>
#include <Kokkos_Core.hpp>

#include "utils.hpp"

#include <iostream>
#include <string>
#include <sstream>

template <typename ExecSpace, int Dim, class Coordinates>
struct Cluster
{
    ArborX::Point<Dim, Coordinates> center;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> points;
    std::string to_string(void) const;
};

template<typename ExecSpace, int Dim, class Coordinates>
inline std::string Cluster<ExecSpace, Dim, Coordinates>::to_string(void) const
{
    std::ostringstream strs;
    strs << "center: " << point_to_str(this->center) << std::endl;
    auto points_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);
    strs << "points:\n";
    for (size_t i = 0; i < points_h.extent(0); ++i) {
        strs << "    - " << point_to_str(points_h(i)) << "\n";
    }
    std::string s = strs.str();
    s.pop_back();
    return s;
}

template<typename ExecSpace, int Dim, class Coordinates>
std::ostream& operator<<(std::ostream& os, Cluster<ExecSpace, Dim, Coordinates> const& cluster)
{
    return os << cluster.to_string();
}

#endif /* ! CLUSTER_HPP */
