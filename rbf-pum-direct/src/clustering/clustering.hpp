#pragma once

#include "callbacks.hpp"
#include "interpolator.hxx"
#include "utils/operators.hpp"

template <int X, int N>
constexpr size_t constexpr_pow()
{
    size_t prod = 1;
    for (size_t i = 0; i < N; ++i)
    {
        prod *= X;
    }
    return prod;
}

template <int Dim>
constexpr auto zCurveNeighborOffsets(size_t shape[Dim])
{
    constexpr size_t N = constexpr_pow<3, Dim>() - 1;
    std::array<int, N> result{};

    std::array<int, Dim> offset{};
    size_t out = 0;

    const size_t total = N + 1;

    for (size_t i = 0; i < total; ++i)
    {
        size_t t = i;
        bool all_zero = true;

        for (int d = 0; d < Dim; ++d)
        {
            int digit = static_cast<int>(t % 3);
            t /= 3;
            digit -= 1;
            offset[d] = digit;
            if (digit != 0)
            {
                all_zero = false;
            }
        }

        if (!all_zero)
        {
            // Indice zCurve du offset calculé
            int index = 0;
            size_t stride = 1;
            for (int k = Dim - 1; k >= 0; --k)
            {
                index += offset[k] * stride;
                stride *= shape[k];
            }
            result[out++] = index;
        }
    }
    return result;
}

template <typename value_type>
struct remove_tagged
{
    KOKKOS_INLINE_FUNCTION
    bool operator()(const value_type& v) const
    {
        return v[0] != v[0];
    }
};

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::create_clusters(void)
{
    assert(this->_radius > 0);
    assert(this->_relative_overlap > 0 && this->_relative_overlap < 1);

    const std::string _region_name = "RbfPumInterpolator::create_clusters";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};

    // 1. calcul de la BB sur les points source
    const Point lower = _source_bvh.bounds().minCorner();
    const Point upper = _source_bvh.bounds().maxCorner();

    // 2. calcul de la distance maximale entre 2 centres pour garantir l'overlap
    const ScalarType spacing =
        std::sqrt(4.0 / Dim) * this->_radius * (1.0 - this->_relative_overlap);

    size_t shape[Dim];
    ScalarType distances[Dim];
    ScalarType min_distance = std::numeric_limits<ScalarType>::max();
    size_t nb_centers = 1;
    for (int axis = 0; axis < Dim; ++axis)
    {
        double edge_length = upper[axis] - lower[axis];
        edge_length = (edge_length < 0) ? (-edge_length) : (edge_length);

        // 3-1. calcul du nombre de centres effectifs dans chaque direction
        shape[axis] = std::ceil(max(1.0, edge_length / spacing));

        // 4. calcul des distances dx, dy, dz (en d dimensions)
        distances[axis] = edge_length / shape[axis];
        if (distances[axis] < min_distance)
        {
            min_distance = distances[axis];
        }

        // 3-2. On rajoute 1 centre dans chaque direction et on considère comme
        // premier centre le point `lower`
        nb_centers *= ++shape[axis];
    }

    // 5. création des centres candidats
    VectorView<Point> centers_candidates(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::centers_candidates"),
        nb_centers);

    Kokkos::parallel_for(
        _region_name + "::p_for fill centers candidates",
        Kokkos::RangePolicy(execspace, 0, nb_centers),
        KOKKOS_LAMBDA(const size_t i) {
            size_t i_copy = i;
            // a. calcul des coordonnées du point
            Kokkos::Array<int, Dim> coords{};
            for (int axis = Dim - 1; axis >= 0; --axis)
            {
                size_t prod = 1;
                for (int d = axis - 1; d >= 0; --d)
                {
                    prod *= shape[d];
                }
                coords[axis] = i_copy / prod;
                i_copy -= coords[axis] * prod;
            }

            // b. calcul de l'indice zCurve
            size_t index = 0;
            size_t stride = 1;
            for (int k = Dim - 1; k >= 0; --k)
            {
                index += coords[k] * stride;
                stride *= shape[k];
            }

            // c. calcul de la position du point dans la BB
            centers_candidates(index) = Point{ lower };
            for (int axis = 0; axis < Dim; ++axis)
            {
                centers_candidates(index)[axis] +=
                    distances[axis] * coords[axis];
            }
        });
    Kokkos::fence();

    // 6. on tag les clusters qui ne contiennent pas de point source donc ceux
    // pour lesquels: norm2(nearest(center), center) > radius
    TagEmptyCenters<ExecSpace, Dim, ScalarType> tag_empty_centers(
        centers_candidates);
    TagEmptyCentersCallback<ExecSpace, Dim, ScalarType>
        tag_empty_centers_callback(centers_candidates, this->_radius);
    this->_source_bvh.query(execspace, tag_empty_centers,
                            tag_empty_centers_callback);

    // 7. on tag les clusters qui ne contiennent pas de point target
    // donc ceux pour lesquels: norm2(nearest(center), center) > radius
    this->_target_bvh.query(execspace, tag_empty_centers,
                            tag_empty_centers_callback);

    // 8. on projette les centres non taggés sur leur point source respectif le
    // plus proche
    TransformToNearest<ExecSpace, Dim, ScalarType> transform_to_nearest(
        centers_candidates);
    TransformToNearestCallback<ExecSpace, Dim, ScalarType>
        transform_to_nearest_callback(centers_candidates);

    this->_source_bvh.query(execspace, transform_to_nearest,
                            transform_to_nearest_callback);

    // 9. on retire les centres qui sont trop proches les uns des autres,
    // threshold = 0.4 * min(distances)
    ScalarType threshold = 0.4 * min_distance;
    threshold *= threshold;
    auto centers_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                            centers_candidates);
    auto neighbors_offsets = zCurveNeighborOffsets<Dim>(shape);
    for (size_t i = 0; i < centers_host.extent(0); ++i)
    {
        // is not NaN = is not tagged
        if (centers_host(i)[0] == centers_host(i)[0])
        {
            for (auto off : neighbors_offsets)
            {
                if (!off)
                {
                    continue;
                }
                long neighbor_id = i + off;
                if (neighbor_id >= 0 && neighbor_id < centers_host.extent(0))
                {
                    auto center = centers_host(i);
                    auto neighbor = centers_host(neighbor_id);
                    // is not NaN = is not tagged AND is too close
                    if (neighbor[0] == neighbor[0]
                        && squared_difference(center, neighbor) < threshold)
                    {
                        centers_host(i)[0] = NAN;
                        break;
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(centers_candidates, centers_host);

    // 10. on tag les clusters qui ne contiennent pas de point target
    // donc ceux pour lesquels: norm2(nearest(center), center) > radius
    tag_empty_centers = decltype(tag_empty_centers)(centers_candidates);
    tag_empty_centers_callback =
        decltype(tag_empty_centers_callback)(centers_candidates, this->_radius);
    this->_target_bvh.query(execspace, tag_empty_centers,
                            tag_empty_centers_callback);

    // 11. on retire les centres taggés
    const auto end = Kokkos::Experimental::remove_if(
        _region_name + "::remove_if", execspace, centers_candidates,
        remove_tagged<Point>{});
    const int dist = Kokkos::Experimental::distance(
        Kokkos::Experimental::begin(centers_candidates), end);
    Kokkos::resize(centers_candidates, dist);

    // 12. on stocke les centres qui ont passé tous les filtres, ce sont nos
    // centres de clusters
    this->_clusters = decltype(this->_clusters)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->clusters"),
        centers_candidates.extent(0));
    Kokkos::deep_copy(this->_clusters, centers_candidates);

    Kokkos::Profiling::popRegion(); // ! create_clusters
}
