#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_LinearBVH.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <iomanip>
#include <random>

#include "callbacks.hpp"
#include "clustering.hpp"
#include "interpolator.hxx"
#include "polynomial.hpp"
#include "rbf_functions.hpp"
#include "utils.hpp"

FULL_TEMPLATE
TEMPLATED_CLASSNAME::RbfPumInterpolator(PointsView source, PointsView target,
                                        PolynomialType polynomial,
                                        RbfFunctionBasisType rbf_function)
{
    this->_source = PointsView("_source", source.extent(0));
    Kokkos::deep_copy(_source, source);
    this->_target = PointsView("_target", target.extent(0));
    Kokkos::deep_copy(_target, target);
    this->_radius = 0;
    this->_polynomial = polynomial;
    this->_rbf_function = rbf_function;

    for (int d = 0; d < Dim; ++d)
    {
        this->no_data[d] = NAN;
    }

    find_radius();
    this->_rbf_function.set_r_inv(1.0 / this->_radius);
    create_clusters();
    prepare_interpolation();
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::find_radius(void)
{
    assert(this->_nodes_per_cluster > 0);
    const ExecSpace execspace{};

    const size_t N = _source.extent(0);
    const size_t M = _target.extent(0);
    PointsView cloud(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "cloud"),
        N + M);

    const auto source_mirror =
        Kokkos::create_mirror_view_and_copy(execspace, _source);
    const auto target_mirror =
        Kokkos::create_mirror_view_and_copy(execspace, _target);

    Kokkos::parallel_for(
        "fill cloud", Kokkos::RangePolicy(execspace, 0, N + M),
        KOKKOS_LAMBDA(const size_t& i) {
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

    KNearest<ExecSpace, Dim, Coordinates> predicate{ _nodes_per_cluster,
                                                     random_points };
    KNearestCallback<ExecSpace, Dim, Coordinates> callback{ random_points };
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
        KOKKOS_LAMBDA(const size_t& i, Coordinates& lsum) {
            lsum += values(i);
        },
        max_radius_sum);
    this->_radius = std::sqrt(max_radius_sum / this->_clustering_rd_samples);
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::prepare_interpolation(void)
{
    assert(this->_clusters.extent(0) > 0 && this->_clusters.extent(1) > 0);
    const ExecSpace execspace{};
    const size_t N = _clusters.extent(1) - 1;
    const size_t M = _clusters.extent(0);
    const size_t Pn = Dim + 1;
    Kokkos::View<Coordinates***, ExecSpace> all_lhs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "all_lhs"),
        M, N + Pn, N + Pn);
    Kokkos::View<Coordinates**, ExecSpace> all_rhs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "all_rhs"),
        M, N + Pn);
    auto f = this->_rbf_function;
    auto data_function = KOKKOS_LAMBDA(const Point& p)
    {
        return 1.0;
    };
    auto clusters = Kokkos::create_mirror_view_and_copy(execspace, _clusters);
    Kokkos::parallel_for(
        "fill all_lhs rbf values (A)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0, 0, 0 }, { N, N, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            if (j >= i)
            {
                auto rbf_val = f(NDdistance<Dim, Coordinates>(
                    clusters(k, i + 1), clusters(k, j + 1)));
                all_lhs(k, i, j) = rbf_val;
                all_rhs(k, i) = data_function(clusters(k, 1 + i));
            }
        });
    auto p = this->_polynomial;
    Kokkos::parallel_for(
        "fill all_rhs poly values (P/Pt)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0, 0 }, { N, M }),
        KOKKOS_LAMBDA(const size_t& j, const size_t& k) {
            auto poly_values = p(clusters(k, 1 + j));
            for (size_t i = 0; i < Pn; ++i)
            {
                all_lhs(k, N + i, j) = poly_values[i];
                all_lhs(k, j, N + i) = poly_values[i];
            }
        });
    Kokkos::parallel_for(
        "fill all_rhs padding zeroes (0)",
        Kokkos::MDRangePolicy(execspace, { N, N, (size_t)0 },
                              { N + Pn, N + Pn, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            all_lhs(k, i, j) = 0.0;
            all_rhs(k, i) = 0.0;
        });
    Kokkos::fence();

    // TODO: solve for alpha & beta using KokkosKernels::batched*
}

FULL_TEMPLATE
Coordinates TEMPLATED_CLASSNAME::get_radius() const
{
    return this->_radius;
}

#endif /* ! INTERPOLATOR_HPP */
