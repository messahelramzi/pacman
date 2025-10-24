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
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    (this->_project_to_input) ? (this->_create_clusters_proj())
                              : (this->_create_clusters_no_proj());
}

FULL_TEMPLATE void TEMPLATED_CLASSNAME::_create_clusters_no_proj(void)
{
    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::_create_clusters_no_proj");
    ExecSpace execspace{};

    const Coordinates spacing =
        std::sqrt(4.0 / Dim) * this->_radius * (1.0 - this->_relative_overlap);
    const Point lower = _source_bvh.bounds().minCorner();
    const Point upper = _source_bvh.bounds().maxCorner();

    size_t nb_elements[Dim];
    for (int i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
    }

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::_create_clusters_no_proj count centers");
    Kokkos::View<size_t, ExecSpace> nb_valid_points(
        "RbfPumInterpolator::_create_clusters_no_proj::nb_valid_points");
    CountValidClusters<ExecSpace, Dim, Coordinates> predicate(
        spacing, this->_radius, lower, nb_elements);
    CountValidClustersCallback<ExecSpace, Dim, Coordinates> callback{
        nb_valid_points
    };

    this->_source_bvh.query(execspace, predicate, callback);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                 nb_valid_points);

    using FindClustersCentersRet = Kokkos::pair<Point, Point>;
    Kokkos::View<FindClustersCentersRet*, ExecSpace> pairs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_no_proj::pairs"),
        m());

    Kokkos::View<size_t, ExecSpace> id(
        "RbfPumInterpolator::_create_clusters_no_proj::id");
    FindClustersCenters<ExecSpace, Dim, Coordinates> predicate2(
        spacing, this->_radius, lower, nb_elements);
    FindClustersCentersCallback<ExecSpace, Dim, Coordinates> callback2{ pairs,
                                                                        id };
    this->_source_bvh.query(execspace, predicate2, callback2);
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::create_clusters
                                    // count centers

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::_create_clusters_no_proj build clusters view");
    Kokkos::sort(pairs,
                 sort_by_center<ExecSpace, Dim, Coordinates, PolynomialType,
                                RbfFunctionBasisType>());
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_no_proj::offsets"),
        pairs.extent(0));
    auto pairs_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs);
    auto offsets_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                Kokkos::HostSpace{}, offsets);
    int cumsum = 0;
    int elts = 0;
    int max_elts = 0;
    size_t k = 1;
    offsets_h(0) = 0;
    for (size_t i = 1; i <= pairs_h.extent(0); ++i)
    {
        elts++;
        if (i == pairs_h.extent(0) || pairs_h(i - 1).first != pairs_h(i).first)
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

    size_t ext0 = --k;
    size_t ext1 = max_elts + 1;

    Kokkos::View<size_t, ExecSpace> id2(
        "RbfPumInterpolator::_create_clusters_no_proj::id2");
    ClustersView clusters(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_no_proj::clusters"),
        ext0, ext1);
    Kokkos::View<size_t*, ExecSpace> nb_values_per_cluster(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::_create_clusters_no_proj::nb_"
                           "values_per_cluster"),
        ext0);
    Point no_data = Point{ this->no_data };
    Kokkos::parallel_for(
        "RbfPumInterpolator::_create_clusters_no_proj::p_for fill clusters "
        "view",
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

FULL_TEMPLATE void TEMPLATED_CLASSNAME::_create_clusters_proj(void)
{
    Kokkos::Profiling::pushRegion("RbfPumInterpolator::_create_clusters_proj");
    ExecSpace execspace{};

    const Coordinates spacing =
        std::sqrt(4.0 / Dim) * this->_radius * (1.0 - this->_relative_overlap);
    const Point lower = _source_bvh.bounds().minCorner();
    const Point upper = _source_bvh.bounds().maxCorner();

    size_t nb_elements[Dim];
    for (int i = 0; i < Dim; ++i)
    {
        nb_elements[i] = (size_t)((upper[i] - lower[i]) / spacing);
    }

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::_create_clusters_proj count centers");
    Kokkos::View<size_t, ExecSpace> nb_valid_points(
        "RbfPumInterpolator::_create_clusters_proj::nb_valid_points");
    CountValidClusters<ExecSpace, Dim, Coordinates> predicate(
        spacing, this->_radius, lower, nb_elements);
    CountValidClustersCallback<ExecSpace, Dim, Coordinates> callback{
        nb_valid_points
    };

    this->_source_bvh.query(execspace, predicate, callback);

    Kokkos::Profiling::popRegion(); // !
                                    // RbfPumInterpolator::_create_clusters_proj
                                    // count centers

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                 nb_valid_points);

    PointsView centers(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_proj::centers"),
        m());
    Kokkos::View<size_t, ExecSpace> id2(
        "RbfPumInterpolator::_create_clusters_proj::id2");

    ProjectToInput<ExecSpace, Dim, Coordinates> predicate2{ nb_elements, lower,
                                                            spacing };
    ProjectToInputCallback<ExecSpace, Dim, Coordinates> callback2{
        centers, id2, this->_radius
    };

    this->_source_bvh.query(execspace, predicate2, callback2);

    auto new_size =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, id2);
    Kokkos::resize(centers, new_size());
    Kokkos::sort(centers);
    auto last = Kokkos::Experimental::unique(execspace, centers);
    Kokkos::resize(centers,
                   centers.extent(0)
                       - Kokkos::Experimental::distance(
                           last, Kokkos::Experimental::end(centers)));

    using CreateClustersPairsRet = Kokkos::pair<Point, Point>;
    Kokkos::View<CreateClustersPairsRet*, ExecSpace> values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::_create_clusters_proj::values"),
        0);
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_proj::offsets"),
        0);
    CreateClustersPairs<ExecSpace, Dim, Coordinates> predicate3{
        centers, this->_radius
    };
    CreateClustersPairsCallback<ExecSpace, Dim, Coordinates> callback3{};

    this->_source_bvh.query(execspace, predicate3, callback3, values, offsets);

    size_t ext0, ext1;
    Kokkos::parallel_reduce(
        "RbfPumInterpolator::_create_clusters_proj::find clusters extents size",
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

    ext1++;
    Kokkos::View<size_t, ExecSpace> id3(
        "RbfPumInterpolator::_create_clusters_proj::id3");
    ClustersView clusters(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::_create_clusters_proj::clusters"),
        ext0, ext1);
    Kokkos::View<size_t*, ExecSpace> nb_values_per_cluster(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::_create_clusters_proj::nb_"
                           "values_per_cluster"),
        ext0);
    Point no_data = Point{ this->no_data };
    Kokkos::parallel_for(
        "RbfPumInterpolator::_create_clusters_proj::p_for fill clusters "
        "view",
        Kokkos::RangePolicy(execspace, 0, offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i) {
            if (offsets(i + 1) - offsets(i) > 0)
            {
                const size_t iid = Kokkos::atomic_fetch_inc(&(id3()));
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

    Kokkos::Profiling::popRegion(); // !
                                    // RbfPumInterpolator::_create_clusters_proj
}

#endif /* ! CLUSTERING_HPP */
