#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>
#include <random>

#include "callbacks.hpp"
#include "cluster.hpp"

template <typename ExecSpace, int Dim = 3, class Coordinates = double>
class RbfPumInterpolator
{
    using Point = ArborX::Point<Dim, Coordinates>;
    using Box = ArborX::Box<Dim, Coordinates>;
    using PointsView = Kokkos::View<Point*, ExecSpace>;
    using ClustersView = Kokkos::View<Cluster<ExecSpace, Dim, Coordinates>*, ExecSpace>;

public:
    RbfPumInterpolator(PointsView source, PointsView target);
    void create_clusters();
    void find_radius();
    Coordinates get_radius() const;
    ClustersView _clusters;

private:
    double _radius;
    const int _nodes_per_cluster = 25;
    const double _relative_overlap = 0.15;
    const size_t _clustering_rd_samples = 100;
    Box _bd;
    PointsView _source;
    PointsView _target;
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
    create_clusters();
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
        Kokkos::RangePolicy(execspace, (size_t)(values.extent(0) * 0.95),
                            values.extent(0)),
        KOKKOS_LAMBDA(const size_t i, Coordinates& lsum) { lsum += values(i); },
        max_radius_sum);
    this->_radius = std::sqrt(max_radius_sum / this->_clustering_rd_samples);
}

template <typename ExecSpace, int Dim, class Coordinates>
void RbfPumInterpolator<ExecSpace, Dim, Coordinates>::create_clusters()
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

    PointsView values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "values"),
        0);
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "offsets"),
        0);
    RemoveEmptyClusters<ExecSpace, Dim, Coordinates> predicate{
        this->_radius, centers_candidates
    };
    RemoveEmptyClustersCallback<ExecSpace, Dim, Coordinates> callback{};
    ArborX::BoundingVolumeHierarchy bvh{ execspace, this->_source };
    ArborX::query(bvh, execspace, predicate, callback, values, offsets);

    // auto hostspace = Kokkos::HostSpace{};
    // auto values_mirror =
    //     Kokkos::create_mirror_view_and_copy(hostspace, values);
    // auto offsets_mirror =
    //     Kokkos::create_mirror_view_and_copy(hostspace, offsets);

    // for (size_t i = 0; i < nb_centers; ++i)
    // {
    //     for (size_t j = offsets_mirror(i); j < offsets_mirror(i + 1); ++j)
    //     {
    //         std::cout << i << ": " << point_to_str(values_mirror(j))
    //                   << std::endl;
    //     }
    // }
    // std::cout << "nb_centers:" << nb_centers << std::endl;

    Kokkos::View<int*, ExecSpace> nb_nodes_per_cluster(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "nb_nodes_per_cluster"),
        nb_centers);
    size_t nb_non_empty_clusters = 0;
    Kokkos::parallel_reduce(
        "count non empty clusters",
        Kokkos::RangePolicy(execspace, 0, nb_centers),
        KOKKOS_LAMBDA(const size_t i, size_t &lval) {
            nb_nodes_per_cluster(i) = 0;
            if (offsets(i) != offsets(i+1)) { lval++; }
            for (size_t j = offsets(i); j < offsets(i + 1); ++j)
            {
                Kokkos::atomic_inc(&(nb_nodes_per_cluster(i)));
            }
        }, nb_non_empty_clusters);

    ClustersView clusters(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "clusters"), nb_centers);
    auto clusters_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, clusters);
    auto nb_nodes_per_cluster_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nb_nodes_per_cluster);
    for (size_t it = 0; it < clusters_h.extent(0); ++it) {
        if (nb_nodes_per_cluster_h(it) > 0) {
            Kokkos::resize(clusters_h(it).points, nb_nodes_per_cluster_h(it));
        }
    }
    Kokkos::deep_copy(clusters, clusters_h);

    Kokkos::parallel_for("fill clusters", Kokkos::RangePolicy(execspace, 0, nb_centers), KOKKOS_LAMBDA(const size_t i) {
        if (nb_nodes_per_cluster(i) != 0) {
            clusters(i).center= centers_candidates(i);
            for (size_t ii = offsets(i); ii < offsets(i+1); ++ii) {
                clusters(i).points(ii - offsets(i)) = Point(values(ii));
            }
        }
    });
    Kokkos::fence();

    clusters_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, clusters);
    size_t reindex_i = 0;
    for (size_t i = 0; i < clusters_h.extent(0); ++i) {
        while (reindex_i < clusters_h.extent(0) && clusters_h(reindex_i).points.extent(0) == 0) {
            reindex_i++;
        }
        if (clusters_h(i).points.extent(0) == 0) {
            std::swap(clusters_h(i), clusters_h(reindex_i));
        }
    }

    for (size_t i = 0;  i < clusters_h.extent(0); ++i) {
        std::cout << clusters_h(i) << std::endl;
    }

    _clusters = ClustersView(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "_clusters"), nb_non_empty_clusters);
    Kokkos::deep_copy(_clusters, clusters_h);
}

template <typename ExecSpace, int Dim, class Coordinates>
Coordinates RbfPumInterpolator<ExecSpace, Dim, Coordinates>::get_radius() const
{
    return this->_radius;
}

#endif /* ! INTERPOLATOR_HPP */
