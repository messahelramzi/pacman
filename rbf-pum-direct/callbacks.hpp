#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include "utils.hpp"

template <typename ExecSpace, int Dim, class Coordinates>
struct DistanceToKNearest
{
    const int _k;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _samples;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<DistanceToKNearest<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const DistanceToKNearest<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._samples.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const DistanceToKNearest<ExecSpace, Dim, Coordinates>& obj,
        std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(obj._samples(i), obj._k),
                              obj._samples(i));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct DistanceToKNearestCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(_NDdistance_without_sqrt<Dim, Coordinates>(
            ArborX::getData(predicate), value));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct GetNonEmptyClusters
{
    Coordinates _spacing;
    size_t _shape[Dim];
    ArborX::Point<Dim, Coordinates> _lower;
    Coordinates _spacing_t[Dim];
    GetNonEmptyClusters(Coordinates spacing,
                        ArborX::Point<Dim, Coordinates> lower,
                        ArborX::Point<Dim, Coordinates> upper,
                        size_t shape[Dim])
        : _spacing(spacing)
        , _lower(lower)
    {
        for (int axis = 0; axis < Dim; ++axis)
        {
            _shape[axis] = shape[axis];
            this->_spacing_t[axis] = (upper[axis] - lower[axis]) / shape[axis];
            // clang-format off
            // +1 at each dimension to compensate for "startGridAtEdge @ precice::CreateClustering.hpp:L399~404"
            // clang-format on
            this->_shape[axis]++;
        }
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<GetNonEmptyClusters<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION auto
    size(const GetNonEmptyClusters<ExecSpace, Dim, Coordinates>& obj)
    {
        size_t prod = 1;
        for (int axis = 0; axis < Dim; ++axis)
        {
            prod *= obj._shape[axis];
        }
        return prod;
    }

    static KOKKOS_FUNCTION auto
    get(const GetNonEmptyClusters<ExecSpace, Dim, Coordinates>& obj, size_t i)
    {
        // const size_t i_copy = i;
        ArborX::Point<Dim, Coordinates> candidate;
        for (int axis = Dim - 1; axis >= 0; --axis)
        {
            candidate[axis] = obj._lower[axis]
                + obj._spacing_t[axis] * (i % obj._shape[axis]);
            i /= obj._shape[axis];
        }
        // Kokkos::printf("%lu. candidate = (%f, %f, %f)\n", i_copy,
        // candidate[0],
        //                candidate[1], candidate[2]);
        return ArborX::attach(ArborX::nearest(candidate, 1), candidate);
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct GetNonEmptyClustersCallbackFirstPass
{
    Coordinates _squared_radius;
    Kokkos::View<size_t, ExecSpace> _count;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        auto distance = _NDdistance_without_sqrt<Dim, Coordinates>(
            ArborX::getData(predicate), value);
        if (distance < 0.0 || distance > this->_squared_radius)
        {
            return;
        }
        Kokkos::atomic_inc(&(this->_count()));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct GetNonEmptyClustersCallbackSecondPass
{
    using ReturnType = Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                                    ArborX::Point<Dim, Coordinates>>;
    Coordinates _squared_radius;
    Kokkos::View<ReturnType*, ExecSpace> _out;
    Kokkos::View<size_t, ExecSpace> _id;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        auto distance = _NDdistance_without_sqrt<Dim, Coordinates>(
            ArborX::getData(predicate), value);
        if (distance < 0.0 || distance > this->_squared_radius)
        {
            return;
        }
        const size_t index = Kokkos::atomic_fetch_inc(&(this->_id()));
        this->_out(index) =
            Kokkos::make_pair(ArborX::getData(predicate), value);
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FilterEmptyClusters
{
    Kokkos::View<Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                              ArborX::Point<Dim, Coordinates>>*,
                 ExecSpace>
        _centers_candidates;
    bool _use_proj;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<FilterEmptyClusters<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION auto
    size(const FilterEmptyClusters<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._centers_candidates.extent(0);
    }

    static KOKKOS_FUNCTION auto
    get(const FilterEmptyClusters<ExecSpace, Dim, Coordinates>& obj, size_t i)
    {
        if (obj._use_proj)
        {
            return ArborX::attach(
                ArborX::nearest(obj._centers_candidates(i).first, 1),
                obj._centers_candidates(i).second);
        }
        else
        {
            return ArborX::attach(
                ArborX::nearest(obj._centers_candidates(i).first, 1),
                obj._centers_candidates(i).first);
        }
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FilterEmptyClustersCallback
{
    using ReturnType = Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                                    ArborX::Point<Dim, Coordinates>>;
    Coordinates _squared_radius;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _out;
    Kokkos::View<size_t, ExecSpace> _id;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        auto distance = _NDdistance_without_sqrt<Dim, Coordinates>(
            ArborX::getData(predicate), value);
        if (distance < 0.0 || distance > this->_squared_radius)
        {
            return;
        }
        const size_t index = Kokkos::atomic_fetch_inc(&(this->_id()));
        this->_out(index) = ArborX::getData(predicate);
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct RemoveDuplicates
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _centers;
    Coordinates _threshold;
    RemoveDuplicates(
        Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace>& centers,
        Coordinates distances[Dim])
        : _centers(centers)
    {
        Coordinates min = std::numeric_limits<Coordinates>::max();
        for (int axis = 0; axis < Dim; ++axis)
        {
            if (distances[axis] < min)
            {
                min = distances[axis];
            }
        }
        // clang-format off
        // threshold formula from precice::CreateClustering.hpp:L478
        // clang-format on
        this->_threshold = 0.4 * min;
        std::cout << "threshold = " << std::setprecision(17) << this->_threshold
                  << std::endl;
        // To avoid sqrt during comparisons
        this->_threshold *= this->_threshold;
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<RemoveDuplicates<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION size_t
    size(const RemoveDuplicates<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const RemoveDuplicates<ExecSpace, Dim, Coordinates>& obj, size_t i)
    {
        return ArborX::attach(ArborX::nearest(obj._centers(i), 2), i);
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct RemoveDuplicatesCallback
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> centers;
    Kokkos::View<bool*, ExecSpace> destroy_list;
    Coordinates _threshold;
    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        const Coordinates distance = _NDdistance_without_sqrt<Dim, Coordinates>(
            centers(ArborX::getData(predicate)), value);
        if (distance > 0.0 && distance < this->_threshold)
        {
            Kokkos::atomic_store(&(destroy_list(ArborX::getData(predicate))),
                                 true);
        }
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct Projection
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _centers;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<Projection<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const Projection<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const Projection<ExecSpace, Dim, Coordinates>& obj, std::size_t i)
    {
        return ArborX::attach(ArborX::nearest(obj._centers(i), 1),
                              obj._centers(i));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ProjectionCallback
{
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        // <center, projection>
        out(Kokkos::make_pair<ArborX::Point<Dim, Coordinates>,
                              ArborX::Point<Dim, Coordinates>>(
            ArborX::getData(predicate), value));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct GetClustersValues
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _centers;
    Coordinates _radius;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<GetClustersValues<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION size_t
    size(const GetClustersValues<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const GetClustersValues<ExecSpace, Dim, Coordinates>& obj, size_t i)
    {
        if (obj._centers(i)[0] != obj._centers(i)[0])
        {
            auto center = obj._centers(i);
            center[0] = 0.0;
            return ArborX::attach(
                ArborX::intersects(
                    ArborX::Sphere<Dim, Coordinates>(center, -1000.0)),
                obj._centers(i));
        }
        return ArborX::attach(
            ArborX::intersects(
                ArborX::Sphere<Dim, Coordinates>(obj._centers(i), obj._radius)),
            obj._centers(i));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct GetClustersValuesCallback
{
    using ReturnType = Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                                    ArborX::Point<Dim, Coordinates>>;
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate predicate, Value const& value,
                                    OutputFunctor const out) const
    {
        const ArborX::Point<Dim, Coordinates> center =
            ArborX::getData(predicate);
        out(ReturnType{ center, value });
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FilterClustersPostProcess
{
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _centers;
};

template <typename ExecSpace, int Dim, class Coordinates>
struct ArborX::AccessTraits<
    FilterClustersPostProcess<ExecSpace, Dim, Coordinates>>
{
    using memory_space = ExecSpace::memory_space;
    static KOKKOS_FUNCTION size_t
    size(const FilterClustersPostProcess<ExecSpace, Dim, Coordinates>& obj)
    {
        return obj._centers.extent(0);
    }
    static KOKKOS_FUNCTION auto
    get(const FilterClustersPostProcess<ExecSpace, Dim, Coordinates>& obj,
        size_t i)
    {
        if (obj._centers(i)[0] != obj._centers(i)[0])
        {
            // clang-format off
            // the center variable is required to avoid modifying the centers (potential kind of race condition)
            // The ArborX::attach call is useless to the process but ensures a consistent return type
            // clang-format on
            auto center = obj._centers(i);
            center[0] = 0.0;
            return ArborX::attach(ArborX::nearest(center, 0), center);
        }
        return ArborX::attach(ArborX::nearest(obj._centers(i), 1),
                              obj._centers(i));
    }
};

template <typename ExecSpace, int Dim, class Coordinates>
struct FilterClustersPostProcessCallback
{
    Coordinates _squared_radius;
    Kokkos::View<ArborX::Point<Dim, Coordinates>*, ExecSpace> _out;
    Kokkos::View<size_t, ExecSpace> _id;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate predicate,
                                    Value const& value) const
    {
        auto distance = _NDdistance_without_sqrt<Dim, Coordinates>(
            ArborX::getData(predicate), value);
        if (distance < 0.0 || distance > this->_squared_radius)
        {
            return;
        }
        const size_t index = Kokkos::atomic_fetch_inc(&(this->_id()));
        this->_out(index) = ArborX::getData(predicate);
    }
};

#endif /* ! CALLBACKS_HPP */
