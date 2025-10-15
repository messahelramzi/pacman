#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

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
struct FindClustersCenters
{
    const Coordinates spacing;
    const Coordinates radius;
    const ArborX::Point<Dim, Coordinates> lower;
    size_t shape[Dim];

    FindClustersCenters(Coordinates s, Coordinates r,
                        ArborX::Point<Dim, Coordinates> l, size_t d[Dim])
        : spacing(s)
        , radius(r)
        , lower(l)
    {
        for (size_t i = 0; i < Dim; ++i)
        {
            shape[i] = d[i];
        }
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<FindClustersCenters<ExecSpace, Dim, Coordinates>>
{
    static KOKKOS_FUNCTION std::size_t
    size(const FindClustersCenters<ExecSpace, Dim, Coordinates>& fcc)
    {
        size_t prod = 1;
        for (size_t i = 0; i < Dim; ++i)
        {
            prod *= fcc.shape[i];
        }
        return prod;
    }

    static KOKKOS_FUNCTION auto
    get(const FindClustersCenters<ExecSpace, Dim, Coordinates>& fcc,
        std::size_t i)
    {
        ArborX::Point<Dim, Coordinates> center;
        for (size_t axis = 0; axis < Dim; ++axis)
        {
            size_t prod = 1;
            for (size_t tt = axis + 1; tt < Dim; ++tt)
            {
                prod *= fcc.shape[tt];
            }
            center[axis] = fcc.lower[axis] + i / prod * fcc.spacing;
            i = i % prod;
        }
        return ArborX::intersects(
            ArborX::Sphere<Dim, Coordinates>(center, fcc.radius));
    }
    using memory_space = ExecSpace::memory_space;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FindClustersCentersCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(value);
    }
};

#endif /* ! CALLBACKS_HPP */
