#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_LinearBVH.hpp>
#include <random>

#include "callbacks.hpp"
#include "clustering.hpp"
#include "interpolator.hxx"

template <typename ExecSpace, int Dim, class Coordinates>
RbfPumInterpolator<ExecSpace, Dim, Coordinates>::RbfPumInterpolator(
    PointsView source, PointsView target)
{
    _source = PointsView("_source", source.extent(0));
    Kokkos::deep_copy(_source, source);
    _target = PointsView("_target", target.extent(0));
    Kokkos::deep_copy(_target, target);
    _radius = 0;

    for (size_t d = 0; d < Dim; ++d)
    {
        no_data[d] = NAN;
    }

    find_radius();
    create_clusters();
}

template <typename ExecSpace, int Dim, class Coordinates>
void RbfPumInterpolator<ExecSpace, Dim, Coordinates>::find_radius(void)
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
        "fill cloud", Kokkos::RangePolicy(execspace, 0, N + M),
        KOKKOS_LAMBDA(const size_t i) {
            cloud(i) = (i < N) ? source_mirror(i) : target_mirror(i - N);
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

    PointsView random_points(Kokkos::view_alloc(execspace,
                                                Kokkos::WithoutInitializing,
                                                "random_points"),
                             this->_clustering_rd_samples);
    auto random_points_mirror =
        Kokkos::create_mirror_view(Kokkos::HostSpace{}, random_points);
    for (size_t i = 0; i < this->_clustering_rd_samples; ++i)
    {
        Point random_point{};
        for (int d = 0; d < Dim; d++)
        {
            random_point[d] = unif_array[d](re_array[d]);
        }
        random_points_mirror(i) = random_point;
    }
    Kokkos::deep_copy(random_points, random_points_mirror);

    KNearest<Dim, Coordinates> predicate{ _nodes_per_cluster, random_points };
    KNearestCallback<Dim, Coordinates> callback{ random_points };
    Kokkos::View<Coordinates*, ExecSpace> values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "values"),
        0);
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "offsets"),
        0);
    ArborX::query(bvh, execspace, predicate, callback, values, offsets);

    Kokkos::sort(values);
    Coordinates max_radius_sum = 0.;
    Kokkos::parallel_reduce(
        "sum max radius",
        Kokkos::RangePolicy(execspace, (size_t)(values.extent(0) * 0.95),
                            values.extent(0)),
        KOKKOS_LAMBDA(const size_t i, Coordinates& lsum) { lsum += values(i); },
        max_radius_sum);
    this->_radius = std::sqrt(max_radius_sum / this->_clustering_rd_samples);
}

template <typename ExecSpace, int Dim, class Coordinates>
Coordinates RbfPumInterpolator<ExecSpace, Dim, Coordinates>::get_radius() const
{
    return this->_radius;
}

#endif /* ! INTERPOLATOR_HPP */
