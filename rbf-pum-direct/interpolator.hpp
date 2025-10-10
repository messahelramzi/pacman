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
    Coordinates get_radius() const;

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
        "fill cloud", Kokkos::RangePolicy(execspace, 0, N + M),
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
        Kokkos::RangePolicy(execspace, (size_t)(values.extent(0) * 0.90),
                            values.extent(0)),
        KOKKOS_LAMBDA(const size_t i, Coordinates& lsum) { lsum += values(i); },
        max_radius_sum);
    this->_radius = std::sqrt(max_radius_sum / this->_clustering_rd_samples);
}

template <typename ExecSpace, int Dim, class Coordinates>
void RbfPumInterpolator<ExecSpace, Dim, Coordinates>::create_centers()
{
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    ExecSpace execspace{};

    const Coordinates spacing =
        (1.0 - this->_relative_overlap) * (2.0 * this->_radius);
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    size_t nb_elements[Dim];
    size_t nb_centers = 1;
    for (size_t i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
        nb_centers *= nb_elements[i];
    }

    PointsView centers_candidates(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "centers_candidates"),
        nb_centers);
    switch (Dim)
    {
    case 1:
        Kokkos::parallel_for(
            "fill centers candidates 1d",
            Kokkos::RangePolicy(execspace, 0, nb_elements[0]),
            KOKKOS_LAMBDA(const size_t i) {
                centers_candidates(i) = Point({ lower[0] + i * spacing });
            });
        break;
    case 2:
        Kokkos::parallel_for(
            "fill centers candidates 2d",
            Kokkos::MDRangePolicy(execspace, { 0, 0 },
                                  { nb_elements[0], nb_elements[1] }),
            KOKKOS_LAMBDA(const size_t i, const size_t j) {
                centers_candidates(i + j * nb_elements[0]) =
                    Point({ lower[0] + i * spacing, lower[1] + j * spacing });
            });
        break;
    case 3:
        Kokkos::parallel_for(
            "fill centers candidates 3d",
            Kokkos::MDRangePolicy(
                execspace, { 0, 0, 0 },
                { nb_elements[0], nb_elements[1], nb_elements[2] }),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                centers_candidates(i + j * nb_elements[0]
                                   + k * nb_elements[0] * nb_elements[1]) =
                    Point({ lower[0] + i * spacing, lower[1] + j * spacing,
                            lower[2] + k * spacing });
            });
        break;
    default:
        Kokkos::abort("RbfPumIntepolator::create_centers: Dim > 3 is not "
                      "implemented yet!");
    }
    Kokkos::fence();
}

template <typename ExecSpace, int Dim, class Coordinates>
Coordinates RbfPumInterpolator<ExecSpace, Dim, Coordinates>::get_radius() const
{
    return this->_radius;
}

#endif /* ! INTERPOLATOR_HPP */
