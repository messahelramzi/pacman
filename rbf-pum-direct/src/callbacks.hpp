#pragma once

#include <ArborX.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include "concepts.hpp"
#include "utils/operators.hpp"

template <KokkosViewRank<1> ViewType>
struct DistanceToKNearest
{
    const int k;
    ViewType samples;
};

template <KokkosViewRank<1> ViewType>
struct ArborX::AccessTraits<DistanceToKNearest<ViewType>>
{
    using memory_space = typename ViewType::memory_space;
    using Self = DistanceToKNearest<ViewType>;
    static KOKKOS_FUNCTION size_t size(const Self& self)
    {
        return self.samples.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const Self& self, size_t i)
    {
        return ArborX::attach(ArborX::nearest(self.samples(i), self.k),
                              self.samples(i));
    }
};

struct DistanceToKNearestCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(squared_difference(ArborX::getData(predicate), value.value));
    }
};

template <KokkosViewRank<1> ViewType>
struct Projection
{
    ViewType centers;
};

template <KokkosViewRank<1> ViewType>
struct ArborX::AccessTraits<Projection<ViewType>>
{
    using memory_space = typename ViewType::memory_space;
    using Self = Projection<ViewType>;
    static KOKKOS_FUNCTION size_t size(const Self& self)
    {
        return self.centers.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const Self& self, size_t i)
    {
        return ArborX::attach(ArborX::nearest(self.centers(i), 1),
                              self.centers(i));
    }
};

struct ProjectionCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        // <centre, projection>
        out(Kokkos::make_pair(ArborX::getData(predicate), value.value));
    }
};

template <KokkosViewRank<1> ViewType>
struct TagEmptyCenters
{
    ViewType centers_candidates;
};

template <KokkosViewRank<1> ViewType>
struct ArborX::AccessTraits<TagEmptyCenters<ViewType>>
{
    using memory_space = typename ViewType::memory_space;
    using Self = TagEmptyCenters<ViewType>;
    using Point = typename ViewType::non_const_value_type;
    static KOKKOS_FUNCTION size_t size(const Self& self)
    {
        return self.centers_candidates.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const Self& self, size_t i)
    {
        // isNaN
        if (self.centers_candidates(i)[0] != self.centers_candidates(i)[0])
        {
            return ArborX::attach(ArborX::nearest(Point{}, 0), -1);
        }
        return ArborX::attach(ArborX::nearest(self.centers_candidates(i), 1),
                              (int)i);
    }
};

template <KokkosViewRank<1> ViewType, typename RbfPumFPType>
struct TagEmptyCentersCallback
{
    ViewType centers_candidates;
    RbfPumFPType threshold;

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

template <KokkosViewRank<1> ViewType>
struct TransformToNearest
{
    ViewType centers_candidates;
};

template <KokkosViewRank<1> ViewType>
struct ArborX::AccessTraits<TransformToNearest<ViewType>>
{
    using memory_space = typename ViewType::memory_space;
    using Self = TransformToNearest<ViewType>;
    using Point = typename ViewType::non_const_value_type;
    static KOKKOS_FUNCTION size_t size(const Self& self)
    {
        return self.centers_candidates.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const Self& self, size_t i)
    {
        // isNaN
        if (self.centers_candidates(i)[0] != self.centers_candidates(i)[0])
        {
            return ArborX::attach(ArborX::nearest(Point{}, 0), -1);
        }
        return ArborX::attach(ArborX::nearest(self.centers_candidates(i), 1),
                              (int)i);
    }
};

template <KokkosViewRank<1> ViewType>
struct TransformToNearestCallback
{
    ViewType centers_candidates;
    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        // The center must be projected to the nearest point (value)
        centers_candidates(ArborX::getData(predicate)) = value.value;
    }
};

template <KokkosViewRank<1> ViewType, typename RbfPumFPType>
struct GetClustersPoints
{
    ViewType centers;
    RbfPumFPType radius;
};

template <KokkosViewRank<1> ViewType, typename RbfPumFPType>
struct ArborX::AccessTraits<GetClustersPoints<ViewType, RbfPumFPType>>
{
    using memory_space = typename ViewType::memory_space;
    using Self = GetClustersPoints<ViewType, RbfPumFPType>;
    static KOKKOS_FUNCTION size_t size(const Self& self)
    {
        return self.centers.extent(0);
    }
    static KOKKOS_FUNCTION auto get(const Self& self, size_t i)
    {
        return ArborX::intersects(ArborX::Sphere(self.centers(i), self.radius));
    }
};

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
