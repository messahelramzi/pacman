#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>
#include <random>

#include "callbacks.hpp"

template <typename ExecSpace, int Dim = 3, class Coordinates = double>
class RbfPumInterpolator
{
    using Point = ArborX::Point<Dim, Coordinates>;
    using Box = ArborX::Box<Dim, Coordinates>;
    using PointsView = Kokkos::View<Point*, ExecSpace>;

public:
    RbfPumInterpolator(PointsView source, PointsView target);
    void create_centers();
    void find_radius();

private:
    double _radius;
    int _nodes_per_cluster = 25;
    double _relative_overlap = 0.15;
    const size_t _clustering_rd_samples = 100;
    Box _bd;
    PointsView _source;
    PointsView _target;
    PointsView _centers;
};

template <typename ExecSpace, int Dim, class Coordinates>
RbfPumInterpolator<ExecSpace, Dim, Coordinates>::RbfPumInterpolator(
    PointsView source, PointsView target)
{
    _source = PointsView("_source", source.extent(0));
    Kokkos::deep_copy(_source, source);
    _target = PointsView("_target", target.extent(0));
    Kokkos::deep_copy(_target, target);
    _radius = 0;

    find_radius();
    // create_centers();
}

template <typename ExecSpace, int Dim, class Coordinates>
void RbfPumInterpolator<ExecSpace, Dim, Coordinates>::find_radius()
{
    assert(this->_nodes_per_cluster > 0);
    const ExecSpace execspace{};

    const size_t N = _source.extent(0);
    const size_t M = _target.extent(0);
    PointsView cloud(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "cloud"),
        N + M);

    auto source_mirror =
        Kokkos::create_mirror_view_and_copy(execspace, _source);
    auto target_mirror =
        Kokkos::create_mirror_view_and_copy(execspace, _target);

    Kokkos::parallel_for(
        "fill-cloud", Kokkos::RangePolicy(execspace, 0, N + M),
        KOKKOS_LAMBDA(const size_t i) {
            cloud(i) = (i < N) ? source_mirror(i) : target_mirror(i);
        });
    Kokkos::fence();

    ArborX::BoundingVolumeHierarchy bvh{
        execspace, ArborX::Experimental::attach_indices(cloud)
    };
    _bd = bvh.bounds();
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    Coordinates coordinates[2 * Dim];
    for (int i = 0; i < Dim; i++)
    {
        coordinates[i] = lower[i];
        coordinates[Dim + i] = upper[i];
    }

    std::default_random_engine re_array[Dim];
    std::uniform_real_distribution<Coordinates> unif_array[Dim];
    for (int i = 0; i < Dim; i++)
    {
        re_array[i] = {};
        unif_array[i] = std::uniform_real_distribution<Coordinates>(
            coordinates[i], coordinates[Dim + i]);
    }

    Coordinates max_radius_sum = 0.;
    for (size_t i = 0; i < _clustering_rd_samples; i++)
    {
        Point random_point{};
        for (int d = 0; d < Dim; d++)
        {
            random_point[d] = unif_array[d](re_array[d]);
        }

        Kokkos::View<Coordinates*, ExecSpace> values(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "values"),
            0);
        Kokkos::View<int*, ExecSpace> offsets(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "offsets"),
            0);

        KNearest<Dim, Coordinates> predicate{ this->_nodes_per_cluster,
                                              random_point };
        KNearestCallback<Dim, Coordinates> callback{ random_point };

        // TODO: combine into one query only
        ArborX::query(bvh, execspace, predicate, callback, values, offsets);

        const auto offsets_mirror =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);

        Coordinates local_max_radius = 0.;
        Kokkos::Max<Coordinates> max_reducer(local_max_radius);
        Kokkos::parallel_reduce(
            "reduce local max radius",
            Kokkos::RangePolicy(execspace, offsets_mirror(0),
                                offsets_mirror(1)),
            KOKKOS_LAMBDA(const size_t i, Coordinates& lmax) {
                max_reducer.join(lmax, values(i));
            },
            max_reducer);
        // Kokkos::fence();
        max_radius_sum += local_max_radius;
    }

    this->_radius = std::sqrt(max_radius_sum / _clustering_rd_samples);
}

#endif /* ! INTERPOLATOR_HPP */
