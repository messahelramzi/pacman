#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include "callbacks.hpp"
#include "interpolator.hxx"

template <typename ExecSpace, int Dim, class Coordinates>
void RbfPumInterpolator<ExecSpace, Dim, Coordinates>::create_clusters(void)
{
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    ExecSpace execspace{};

    const Coordinates spacing =
        (1.0 - this->_relative_overlap) * (2.0 * this->_radius);
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    size_t nb_elements[Dim];
    for (size_t i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
    }

    FindClustersCenters<ExecSpace, Dim, Coordinates> predicate(
        spacing, this->_radius, lower, nb_elements);
    FindClustersCentersCallback<ExecSpace, Dim, Coordinates> callback;
    Kokkos::View<Point*, ExecSpace> values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "values"),
        0);

    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "offsets"),
        0);
    ArborX::BoundingVolumeHierarchy bvh{ execspace, this->_source };
    ArborX::query(bvh, execspace, predicate, callback, values, offsets);

    size_t N = values.extent(0);
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
    Kokkos::View<size_t, ExecSpace> id("id");
    ClustersView clusters(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "clusters"),
        ext0, ext1);
    Kokkos::parallel_for(
        "fill clusters view",
        Kokkos::RangePolicy(execspace, 0, offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i) {
            if (offsets(i + 1) - offsets(i) > 0)
            {
                const size_t iid = Kokkos::atomic_fetch_inc(&(id()));
                for (size_t ii = offsets(i); ii < offsets(i + 1); ++ii)
                {
                    clusters(iid, ii - offsets(i)) = values(ii);
                }
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
}

#endif /* ! CLUSTERING_HPP */
