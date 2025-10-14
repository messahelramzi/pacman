#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>

#include "cluster.hpp"
#include "utils.hpp"

template <int Dim, class Coordinates>
struct KNearest
{
    int k;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*> random_points;
};

template <int Dim, class Coordinates>
struct ArborX::AccessTraits<KNearest<Dim, Coordinates>>
{
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const KNearest<Dim, Coordinates>& kn)
    {
        return kn.random_points.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const KNearest<Dim, Coordinates>& kn,
                                    std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(kn.random_points(i), kn.k), i);
    }
};

template <int Dim, class Coordinates>
struct KNearestCallback
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*> centers;
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(_NDdistance_without_sqrt<Dim, Coordinates>(
            centers(ArborX::getData(predicate)), value.value));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct RemoveEmptyClusters
{
    Coordinates radius;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> centers;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<RemoveEmptyClusters<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const RemoveEmptyClusters<ExecSpace, Dim, Coordinates>& rec)
    {
        return rec.centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const RemoveEmptyClusters<ExecSpace, Dim, Coordinates>& rec,
        std::size_t i)
    {
        return ArborX::intersects(
            ArborX::Sphere<Dim, Coordinates>(rec.centers(i), rec.radius));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct RemoveEmptyClustersCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(value);
    }
};
#endif /* ! CALLBACKS_HPP */
