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

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::create_clusters(void)
{
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);
    PRINT_STRING("Entering create_clusters");
    const std::string _region_name = "RbfPumInterpolator::create_clusters";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};

    const Coordinates spacing =
        std::sqrt(4.0 / Dim) * this->_radius * (1.0 - this->_relative_overlap);
    DEBUG_FLOAT(spacing);
    const Point lower = _source_bvh.bounds().minCorner();
    const Point upper = _source_bvh.bounds().maxCorner();

    size_t nb_elements[Dim];
    for (int i = 0; i < Dim; ++i)
    {
        nb_elements[i] = std::ceil((upper[i] - lower[i]) / spacing);
        DEBUG_INT(nb_elements[i]);
    }

    using ReturnType = Kokkos::pair<Point, Point>;
    Kokkos::View<size_t, ExecSpace> nb_matching_with_source(
        _region_name + "::nb_matching_with_source");

    GetNonEmptyClusters<ExecSpace, Dim, Coordinates>
        non_empty_clusters_predicate(spacing, lower, upper, nb_elements);
    GetNonEmptyClustersCallbackFirstPass<ExecSpace, Dim, Coordinates>
        source_non_empty_clusters_callback_first_pass{
            this->_radius * this->_radius, nb_matching_with_source
        };
    this->_source_bvh.query(execspace, non_empty_clusters_predicate,
                            source_non_empty_clusters_callback_first_pass);

    auto N = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                 nb_matching_with_source);
    DEBUG_INT(N());
    Kokkos::View<size_t, ExecSpace> global_source_index(
        _region_name + "::global_source_index");
    Kokkos::View<ReturnType*, ExecSpace> centers_matching_with_source(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::centers_matching_with_source"),
        N());
    GetNonEmptyClustersCallbackSecondPass<ExecSpace, Dim, Coordinates>
        source_non_empty_clusters_callback_second_pass{
            this->_radius * this->_radius, centers_matching_with_source,
            global_source_index
        };
    this->_source_bvh.query(execspace, non_empty_clusters_predicate,
                            source_non_empty_clusters_callback_second_pass);

    Kokkos::View<Point*, ExecSpace> clusters_centers(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::clusters_centers"),
        centers_matching_with_source.extent(0));
    Kokkos::View<size_t, ExecSpace> global_clusters_index(
        _region_name + "::global_clusters_index");
    FilterEmptyClusters<ExecSpace, Dim, Coordinates>
        filter_empty_clusters_predicate{ centers_matching_with_source,
                                         this->_project_to_input };
    FilterEmptyClustersCallback<ExecSpace, Dim, Coordinates>
        filter_empty_clusters_callback{ this->_radius * this->_radius,
                                        clusters_centers,
                                        global_clusters_index };

    this->_target_bvh.query(execspace, filter_empty_clusters_predicate,
                            filter_empty_clusters_callback);

    auto M = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                 global_clusters_index);
    DEBUG_INT(M());
    Kokkos::resize(clusters_centers, M());

    const auto distances = non_empty_clusters_predicate._spacing_t;
    Coordinates min = std::numeric_limits<Coordinates>::max();
    for (int axis = 0; axis < Dim; ++axis)
    {
        if (distances[axis] < min)
        {
            min = distances[axis];
        }
    }
    // Squared to avoid sqrt during comparisons
    const Coordinates threshold = 0.16 * min * min;
    DEBUG_FLOAT(threshold);

    /* We must ensure the sequential order of the comparison/tagging of centers
     * to have a deterministic clustering.
     * Random order of tagging we lead to the removal of different nodes
     * and thus lead to undeterministic clustering.
     */
    auto clusters_centers_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, clusters_centers);
    for (size_t i = 0; i < M(); ++i)
    {
        for (size_t j = i + 1; j < M(); ++j)
        {
            const Coordinates distance = _NDdistance_without_sqrt(
                clusters_centers_host(i), clusters_centers_host(j));
            if (distance > 0.0 && distance < threshold)
            {
                clusters_centers_host(i)[0] = NAN;
            }
        }
    }
    Kokkos::deep_copy(clusters_centers, clusters_centers_host);

    Kokkos::sort(clusters_centers);
    auto last = Kokkos::Experimental::unique(execspace, clusters_centers);
    Kokkos::resize(clusters_centers,
                   clusters_centers.extent(0)
                       - Kokkos::Experimental::distance(
                           last, Kokkos::Experimental::end(clusters_centers)));

    PointsView final_clusters_centers(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::final_clusters_centers"),
        clusters_centers.extent(0));
    Kokkos::View<size_t, ExecSpace> final_clusters_filter_index(
        _region_name + "::final_clusters_filter_index");

    FilterClustersPostProcess<ExecSpace, Dim, Coordinates>
        post_process_predicate{ clusters_centers };
    FilterClustersPostProcessCallback<ExecSpace, Dim, Coordinates>
        post_process_callback{ this->_radius * this->_radius,
                               final_clusters_centers,
                               final_clusters_filter_index };

    this->_target_bvh.query(execspace, post_process_predicate,
                            post_process_callback);

    auto index_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, final_clusters_filter_index);
    Kokkos::resize(final_clusters_centers, index_mirror());

    Kokkos::sort(final_clusters_centers);
    auto last2 =
        Kokkos::Experimental::unique(execspace, final_clusters_centers);
    Kokkos::resize(
        final_clusters_centers,
        final_clusters_centers.extent(0)
            - Kokkos::Experimental::distance(
                last2, Kokkos::Experimental::end(final_clusters_centers)));

    Kokkos::View<ReturnType*, ExecSpace> clusters_values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::clusters_values"),
        0);
    Kokkos::View<int*, ExecSpace> clusters_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::clusters_offsets"),
        0);
    GetClustersValues<ExecSpace, Dim, Coordinates>
        get_clusters_values_predicate{ final_clusters_centers, this->_radius };
    GetClustersValuesCallback<ExecSpace, Dim, Coordinates>
        get_clusters_values_callback{};
    this->_source_bvh.query(execspace, get_clusters_values_predicate,
                            get_clusters_values_callback, clusters_values,
                            clusters_offsets);

    size_t ext0, ext1;
    Kokkos::parallel_reduce(
        _region_name + "::find clusters extents size",
        Kokkos::RangePolicy(execspace, 0, clusters_offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i, size_t& lmax, size_t& lsum) {
            size_t l = clusters_offsets(i + 1) - clusters_offsets(i);
            if (l > 0)
            {
                lsum++;
            }
            lmax = (l > lmax) ? (l) : (lmax);
        },
        Kokkos::Max<size_t>(ext1), ext0);

    ext1++;
    Kokkos::View<size_t, ExecSpace> global_clusters_index_final(
        _region_name + "::global_clusters_index_final");
    ClustersView clusters(Kokkos::view_alloc(execspace,
                                             Kokkos::WithoutInitializing,
                                             _region_name + "::clusters"),
                          ext0, ext1);
    Kokkos::View<size_t*, ExecSpace> nb_values_per_cluster(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::nb_values_per_cluster"),
        ext0);
    Kokkos::parallel_for(
        _region_name + "::p_for fill clusters view",
        Kokkos::RangePolicy(execspace, 0, clusters_offsets.size() - 1),
        KOKKOS_LAMBDA(const size_t i) {
            if (clusters_offsets(i + 1) - clusters_offsets(i) > 0)
            {
                const size_t iid =
                    Kokkos::atomic_fetch_inc(&(global_clusters_index_final()));
                clusters(iid, 0) = clusters_values(clusters_offsets(i)).first;
                for (size_t ii = clusters_offsets(i);
                     ii < clusters_offsets(i + 1); ++ii)
                {
                    clusters(iid, 1 + ii - clusters_offsets(i)) =
                        clusters_values(ii).second;
                }
                nb_values_per_cluster(iid) =
                    clusters_offsets(i + 1) - clusters_offsets(i);
                for (size_t ii = clusters_offsets(i + 1) - clusters_offsets(i);
                     ii < ext1; ++ii)
                {
                    clusters(iid, ii) = ArborX::Point<Dim, Coordinates>{};
                    ;
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
}

#endif /* ! CLUSTERING_HPP */
