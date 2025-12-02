#pragma once

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include "utils/operators.hpp"

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct DistanceToKNearest
{
    const int _k;
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace> _samples;
};

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct ArborX::AccessTraits<DistanceToKNearest<ExecSpace, Dim, RbfPumFPType>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const DistanceToKNearest<ExecSpace, Dim, RbfPumFPType>& obj)
    {
        return obj._samples.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const DistanceToKNearest<ExecSpace, Dim, RbfPumFPType>& obj,
        std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(obj._samples(i), obj._k),
                              obj._samples(i));
    }
};

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct DistanceToKNearestCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(squared_difference<Dim, RbfPumFPType>(ArborX::getData(predicate),
                                                  value.value));
    }
};

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct Projection
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace> centers;
};

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct ArborX::AccessTraits<Projection<ExecSpace, Dim, RbfPumFPType>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const Projection<ExecSpace, Dim, RbfPumFPType>& obj)
    {
        return obj.centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const Projection<ExecSpace, Dim, RbfPumFPType>& obj, std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(obj.centers(i), 1),
                              obj.centers(i));
    }
};

template <typename ExecSpace, int Dim, class RbfPumFPType>
struct ProjectionCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        // <center, projection>
        out(Kokkos::make_pair<ArborX::Point<Dim, RbfPumFPType>,
                              ArborX::Point<Dim, RbfPumFPType>>(
            ArborX::getData(predicate), value.value));
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct TagEmptyCenters
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace>
        centers_candidates;

    TagEmptyCenters(Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace>&
                        _centers_candidates)
        : centers_candidates(_centers_candidates)
    {}
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct ArborX::AccessTraits<TagEmptyCenters<ExecSpace, Dim, RbfPumFPType>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const TagEmptyCenters<ExecSpace, Dim, RbfPumFPType>& obj)
    {
        return obj.centers_candidates.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const TagEmptyCenters<ExecSpace, Dim, RbfPumFPType>& obj, size_t i)
    {
        if (obj.centers_candidates(i)[0]
            != obj.centers_candidates(i)[0]) // isnan / taggé
        {
            return ArborX::attach(
                ArborX::nearest(ArborX::Point<Dim, RbfPumFPType>{}, 0), -1);
        }
        return ArborX::attach(ArborX::nearest(obj.centers_candidates(i), 1),
                              (int)i);
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct TagEmptyCentersCallback
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace>
        centers_candidates;
    RbfPumFPType threshold;

    TagEmptyCentersCallback(Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*,
                                         ExecSpace>& _centers_candidates,
                            const RbfPumFPType _radius)
        : centers_candidates(_centers_candidates)
    {
        threshold = _radius * _radius;
    }

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        const int center_index = ArborX::getData(predicate);
        // The center should be removed
        if (squared_difference(centers_candidates(center_index), value.value)
            > threshold)
        {
            // Only one match per center, so centers_candidates(center_index) is
            // accessed only once
            // No need of atomic operation here
            centers_candidates(center_index)[0] = NAN;
        }
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct TransformToNearest
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace>
        centers_candidates;

    TransformToNearest(Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*,
                                    ExecSpace>& _centers_candidates)
        : centers_candidates(_centers_candidates)
    {}
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct ArborX::AccessTraits<TransformToNearest<ExecSpace, Dim, RbfPumFPType>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const TransformToNearest<ExecSpace, Dim, RbfPumFPType>& obj)
    {
        return obj.centers_candidates.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const TransformToNearest<ExecSpace, Dim, RbfPumFPType>& obj, size_t i)
    {
        if (obj.centers_candidates(i)[0]
            != obj.centers_candidates(i)[0]) // isnan / taggé
        {
            return ArborX::attach(
                ArborX::nearest(ArborX::Point<Dim, RbfPumFPType>{}, 0), -1);
        }
        return ArborX::attach(ArborX::nearest(obj.centers_candidates(i), 1),
                              (int)i);
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct TransformToNearestCallback
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace>
        centers_candidates;
    TransformToNearestCallback(Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*,
                                            ExecSpace>& _centers_candidates)
        : centers_candidates(_centers_candidates)
    {}
    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        // The center must be projected to the nearest point (value)
        centers_candidates(ArborX::getData(predicate)) = value.value;
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct GetClustersPoints
{
    Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace> centers;
    RbfPumFPType radius;

    GetClustersPoints(
        Kokkos::View<ArborX::Point<Dim, RbfPumFPType>*, ExecSpace> _centers,
        const RbfPumFPType _radius)
        : centers(_centers)
        , radius(_radius)
    {}
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct ArborX::AccessTraits<GetClustersPoints<ExecSpace, Dim, RbfPumFPType>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION size_t
    size(const GetClustersPoints<ExecSpace, Dim, RbfPumFPType>& obj)
    {
        return obj.centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const GetClustersPoints<ExecSpace, Dim, RbfPumFPType>& obj, size_t i)
    {
        return ArborX::intersects(
            ArborX::Sphere<Dim, RbfPumFPType>(obj.centers(i), obj.radius));
    }
};

template <typename ExecSpace, int Dim, typename RbfPumFPType>
struct GetClustersPointsCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate /* predicate */,
                                    Value const& value,
                                    OutputFunctor const& out) const
    {
        out(value.index);
    }
};
