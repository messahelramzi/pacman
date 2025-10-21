#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include "utils.hpp"

template <typename ExecSpace, int Dim, class Coordinates>
struct KNearest
{
    const int k;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> random_points;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<KNearest<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const KNearest<ExecSpace, Dim, Coordinates>& kn)
    {
        return kn.random_points.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const KNearest<ExecSpace, Dim, Coordinates>& kn, std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(kn.random_points(i), kn.k), i);
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct KNearestCallback
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> centers;
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(_NDdistance_without_sqrt<Dim, Coordinates>(
            centers(ArborX::getData(predicate)), value.value));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct CountValidClusters
{
    const Coordinates spacing;
    const Coordinates radius;
    const ArborX::Point<Dim, Coordinates> lower;
    size_t shape[Dim];

    CountValidClusters(Coordinates s, Coordinates r,
                       ArborX::Point<Dim, Coordinates> l, size_t d[Dim])
        : spacing(s)
        , radius(r)
        , lower(l)
    {
        for (int i = 0; i < Dim; ++i)
        {
            shape[i] = d[i];
        }
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<CountValidClusters<ExecSpace, Dim, Coordinates>>
{
    static KOKKOS_FUNCTION std::size_t
    size(const CountValidClusters<ExecSpace, Dim, Coordinates>& cvc)
    {
        size_t prod = 1;
        for (int i = 0; i < Dim; ++i)
        {
            prod *= cvc.shape[i];
        }
        return prod;
    }

    static KOKKOS_FUNCTION auto
    get(const CountValidClusters<ExecSpace, Dim, Coordinates>& cvc,
        std::size_t i)
    {
        ArborX::Point<Dim, Coordinates> center;
        for (int axis = 0; axis < Dim; ++axis)
        {
            size_t prod = 1;
            for (int tt = axis + 1; tt < Dim; ++tt)
            {
                prod *= cvc.shape[tt];
            }
            center[axis] = cvc.lower[axis] + i / prod * cvc.spacing;
            i = i % prod;
        }
        return ArborX::intersects(
            ArborX::Sphere<Dim, Coordinates>(center, cvc.radius));
    }
    using memory_space = ExecSpace::memory_space;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct CountValidClustersCallback
{
    Kokkos::View<size_t, ExecSpace> _count;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate, Value const& value) const
    {
        Kokkos::atomic_inc(&(this->_count()));
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
        for (int i = 0; i < Dim; ++i)
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
        for (int i = 0; i < Dim; ++i)
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
        for (int axis = 0; axis < Dim; ++axis)
        {
            size_t prod = 1;
            for (int tt = axis + 1; tt < Dim; ++tt)
            {
                prod *= fcc.shape[tt];
            }
            center[axis] = fcc.lower[axis] + i / prod * fcc.spacing;
            i = i % prod;
        }
        return ArborX::attach(
            ArborX::intersects(
                ArborX::Sphere<Dim, Coordinates>(center, fcc.radius)),
            center);
    }
    using memory_space = ExecSpace::memory_space;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FindClustersCentersCallback
{
    Kokkos::View<Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                              ArborX::Point<Dim, Coordinates>>*,
                 ExecSpace>
        _out;
    Kokkos::View<size_t, ExecSpace> _id;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        size_t idd = Kokkos::atomic_fetch_inc(&(this->_id()));
        this->_out(idd) = Kokkos::make_pair<ArborX::Point<Dim, Coordinates>,
                                            ArborX::Point<Dim, Coordinates>>(
            ArborX::getData(predicate), value);
    }
};

#endif /* ! CALLBACKS_HPP */
