#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include "callbacks.hpp"
#include "interpolator.hxx"
#include "utils.hpp"

FULL_TEMPLATE struct sort_by_center
{
    KOKKOS_FUNCTION constexpr bool
    operator()(const Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                                  ArborX::Point<Dim, Coordinates>>& lhs,
               const Kokkos::pair<ArborX::Point<Dim, Coordinates>,
                                  ArborX::Point<Dim, Coordinates>>& rhs) const
    {
        return _NDdistance_without_sqrt<Dim, Coordinates>(
                   ArborX::Point<Dim, Coordinates>{}, lhs.first)
            < _NDdistance_without_sqrt<Dim, Coordinates>(
                   ArborX::Point<Dim, Coordinates>{}, rhs.first);
    }
};

FULL_TEMPLATE void TEMPLATED_CLASSNAME::create_clusters(void)
{
    Kokkos::Profiling::pushRegion("RbfPumInterpolator::create_clusters");
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    ExecSpace execspace{};

    const Coordinates spacing =
        std::sqrt(4.0 / Dim) * this->_radius * (1.0 - this->_relative_overlap);
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    size_t nb_elements[Dim];
    for (int i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
    }

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::create_clusters count centers");
    Kokkos::View<size_t, ExecSpace> nb_valid_points(
        "RbfPumInterpolator::create_clusters::nb_valid_points");
    CountValidClusters<ExecSpace, Dim, Coordinates> predicate(
        spacing, this->_radius, lower, nb_elements);
    CountValidClustersCallback<ExecSpace, Dim, Coordinates> callback{
        nb_valid_points
    };
    ArborX::BoundingVolumeHierarchy bvh{ execspace, this->_source };

    bvh.query(execspace, predicate, callback);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                 nb_valid_points);

    using FindClustersCentersRet = Kokkos::pair<Point, Point>;
    Kokkos::View<FindClustersCentersRet*, ExecSpace> pairs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::create_clusters::pairs"),
        m());

    Kokkos::View<size_t, ExecSpace> id("id");
    FindClustersCenters<ExecSpace, Dim, Coordinates> predicate2(
        spacing, this->_radius, lower, nb_elements);
    FindClustersCentersCallback<ExecSpace, Dim, Coordinates> callback2{ pairs,
                                                                        id };
    bvh.query(execspace, predicate2, callback2);
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::create_clusters
                                    // count centers

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::create_clusters build clusters view");
    Kokkos::sort(pairs,
                 sort_by_center<ExecSpace, Dim, Coordinates, PolynomialType,
                                RbfFunctionBasisType>());
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::create_clusters::offsets"),
        pairs.extent(0));
    auto pairs_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs);
    auto offsets_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, offsets);
    int cumsum = 0;
    int elts = 0;
    int max_elts = 0;
    size_t k = 1;
    offsets_h(0) = 0;
    for (size_t i = 1; i <= pairs_h.extent(0); ++i)
    {
        elts++;
        if (i == pairs_h.extent(0)
            || !points_are_equal(pairs_h(i - 1).first, pairs_h(i).first))
        {
            cumsum += elts;
            max_elts = (elts > max_elts) ? (elts) : (max_elts);
            elts = 0;
            offsets_h(k++) = cumsum;
        }
    }

    Kokkos::resize(offsets_h, k);
    Kokkos::resize(offsets, k);
    Kokkos::deep_copy(offsets, offsets_h);

    size_t ext0 = k;
    size_t ext1 = max_elts + 1;

    Kokkos::View<size_t, ExecSpace> id2(
        "RbfPumInterpolator::create_clusters::id2");
    ClustersView clusters(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::create_clusters::clusters"),
        ext0, ext1);
    Kokkos::View<size_t*, ExecSpace> nb_values_per_cluster(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::create_clusters::nb_values_per_cluster"),
        ext0);
    Point no_data = Point{ this->no_data };
    Kokkos::parallel_for(
        "RbfPumInterpolator::create_clusters::p_for fill clusters view",
        Kokkos::RangePolicy(execspace, 0, offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i) {
            if (offsets(i + 1) - offsets(i) > 0)
            {
                const size_t iid = Kokkos::atomic_fetch_inc(&(id2()));
                clusters(iid, 0) = pairs(offsets(i)).first;
                for (size_t ii = offsets(i); ii < offsets(i + 1); ++ii)
                {
                    clusters(iid, 1 + ii - offsets(i)) = pairs(ii).second;
                }
                nb_values_per_cluster(iid) = offsets(i + 1) - offsets(i);
                for (size_t ii = offsets(i + 1) - offsets(i); ii < ext1; ++ii)
                {
                    clusters(iid, ii) = no_data;
                }
            }
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::create_clusters
                                    // build clusters view

    _clusters =
        ClustersView(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                                        "this->_clusters"),
                     clusters.extent(0), clusters.extent(1));
    Kokkos::deep_copy(this->_clusters, clusters);
    _nb_values_per_cluster = Kokkos::View<size_t*, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_nb_values_per_cluster"),
        ext0);
    Kokkos::deep_copy(_nb_values_per_cluster, nb_values_per_cluster);
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::create_clusters
}

#endif /* ! CLUSTERING_HPP */
