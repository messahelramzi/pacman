#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include "callbacks.hpp"
#include "interpolator.hxx"
#include "utils.hpp"

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::create_clusters(void)
{
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    ExecSpace execspace{};

    const Coordinates spacing =
        (1.0 - this->_relative_overlap) * (2.0 * this->_radius);
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    size_t nb_elements[Dim];
    for (int i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
    }

    FindClustersCenters<ExecSpace, Dim, Coordinates> predicate(
        spacing, this->_radius, lower, nb_elements);
    FindClustersCentersCallback<ExecSpace, Dim, Coordinates> callback;
    using CallbackRetType = Kokkos::pair<Point, Point>; // <center, value>
    Kokkos::View<CallbackRetType*, ExecSpace> values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "values"),
        0);

    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "offsets"),
        0);
    ArborX::BoundingVolumeHierarchy bvh{ execspace, this->_source };
    ArborX::query(bvh, execspace, predicate, callback, values, offsets);

    const size_t N = values.extent(0);
    size_t ext0 = 0;
    size_t ext1 = 0;
    Point no_data = Point{ this->no_data };
    Kokkos::parallel_reduce(
        "find clusters extents size",
        Kokkos::RangePolicy(execspace, 0, offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i, size_t& lmax, size_t& lsum) {
            size_t l = offsets(i + 1) - offsets(i);
            if (l > 0)
            {
                lsum++;
            }
            lmax = (l > lmax) ? (l) : (lmax);
        },
        Kokkos::Max<size_t>(ext1), ext0);
    ext1++; // room for the center
    Kokkos::View<size_t, ExecSpace> id("id");
    ClustersView clusters(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "clusters"),
        ext0, ext1);
    Kokkos::View<size_t*, ExecSpace> nb_values_per_cluster(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "nb_values_per_cluster"),
        ext0);
    Kokkos::parallel_for(
        "fill clusters view",
        Kokkos::RangePolicy(execspace, 0, offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i) {
            if (offsets(i + 1) - offsets(i) > 0)
            {
                const size_t iid = Kokkos::atomic_fetch_inc(&(id()));
                clusters(iid, 0) = values(offsets(i)).first;
                for (size_t ii = offsets(i); ii < offsets(i + 1); ++ii)
                {
                    clusters(iid, 1 + ii - offsets(i)) = values(ii).second;
                }
                nb_values_per_cluster(iid) = offsets(i + 1) - offsets(i);
                for (size_t ii = offsets(i + 1) - offsets(i); ii < ext1; ++ii)
                {
                    clusters(iid, ii) = no_data;
                }
            }
        });
    Kokkos::fence();
    _clusters = ClustersView(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "_clusters"),
        clusters.extent(0), clusters.extent(1));
    Kokkos::deep_copy(this->_clusters, clusters);
    _nb_values_per_cluster = Kokkos::View<size_t*, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "_nb_values_per_cluster"),
        ext0);
    Kokkos::deep_copy(_nb_values_per_cluster, nb_values_per_cluster);
}

#endif /* ! CLUSTERING_HPP */
