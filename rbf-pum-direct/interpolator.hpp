#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_LinearBVH.hpp>
#include <KokkosBatched_Getrf.hpp>
#include <KokkosBatched_Getrs.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Random.hpp>
#include <iomanip>

#include "callbacks.hpp"
#include "clustering.hpp"
#include "interpolator.hxx"
#include "polynomial.hpp"
#include "rbf_functions.hpp"
#include "utils.hpp"

/* Constructor for the RBF PUM Interpolator:
** Once it returns from the constructor, the interpolator is ready to use, with
*the class member `interpolate_at`.
** The constructor automatically calls this member with the given `target`
*points.
** @param source: a 1-D `Kokkos::View` which contains source points of the
*field.
** @param values: a 1-D `Kokkos::View` which contains the values associated to
*the source points of the field. Its size is the same as `source`'s.
** @param target: a 1-D `Kokkos::View` which contains the points we want to
*interpolate at.
** @param polynomial: A user-defined functor that is used to perform a
*polynomial augmentation on the rbf matrices.
** @param rbf_function: A user-defined functor which is the rbf function used
*for clusters points evaluation.
*/
FULL_TEMPLATE
TEMPLATED_CLASSNAME::RbfPumInterpolator(
    PointsView source, Kokkos::View<Coordinates*, ExecSpace> values,
    PointsView target, PolynomialType polynomial,
    RbfFunctionBasisType rbf_function)
{
    Kokkos::Profiling::pushRegion("RbfPumInterpolator::RbfPumInterpolator");
    const ExecSpace execspace{};
    this->_source =
        PointsView(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                                      "this->_source"),
                   source.extent(0));
    Kokkos::deep_copy(_source, source);
    this->_values = Kokkos::View<Coordinates*, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_values"),
        values.extent(0));
    Kokkos::deep_copy(_values, values);
    this->_target =
        PointsView(Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                                      "this->_target"),
                   target.extent(0));
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
    // prepare_interpolation();
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::RbfPumInterpolator
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::find_radius(void)
{
    Kokkos::Profiling::pushRegion("RbfPumInterpolator::find_radius");
    assert(this->_nodes_per_cluster > 0);
    const ExecSpace execspace{};

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::find_radius build cloud");
    const size_t N = this->_source.extent(0);
    const size_t M = this->_target.extent(0);
    // cloud=[source, target]
    PointsView cloud(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::find_radius::cloud"),
        N + M);

    // source: [0, N[,
    // target: [N, M[,
    // 0x0LU = 0 but with unsigned long type (inference issue)
    auto source_subview = Kokkos::subview(cloud, Kokkos::make_pair(0x0LU, N));
    auto target_subview = Kokkos::subview(cloud, Kokkos::make_pair(N, N + M));

    Kokkos::deep_copy(source_subview, this->_source);
    Kokkos::deep_copy(target_subview, this->_target);

    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::find_radius build
                                    // cloud

    ArborX::BoundingVolumeHierarchy bvh{
        execspace, ArborX::Experimental::attach_indices(cloud)
    };

    _bd = bvh.bounds();
    const Point lower = _bd.minCorner();
    const Point upper = _bd.maxCorner();

    Kokkos::Profiling::pushRegion(
        "RbfPumInterpolator::find_radius generate random points");
    uint32_t _seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift1024_Pool<> random_pool(_seed);

    PointsView random_points(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::find_radius::random_points"),
        this->_clustering_rd_samples);
    Kokkos::parallel_for(
        "RbfPumInterpolator::find_radius::p_for fill random points",
        Kokkos::RangePolicy(execspace, 0, this->_clustering_rd_samples),
        KOKKOS_LAMBDA(const size_t& i) {
            Point rd_point{};
            auto gen = random_pool.get_state();
            for (int d = 0; d < Dim; ++d)
            {
                rd_point[d] = gen.drand(lower[d], upper[d]);
            }
            random_pool.free_state(gen);
            random_points(i) = rd_point;
        });
    Kokkos::fence();

    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::find_radius
                                    // generate random points

    KNearest<ExecSpace, Dim, Coordinates> predicate{ _nodes_per_cluster,
                                                     random_points };
    KNearestCallback<ExecSpace, Dim, Coordinates> callback{ random_points };
    Kokkos::View<Coordinates*, ExecSpace> values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::find_radius::values"),
        0);
    Kokkos::View<int*, ExecSpace> offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::find_radius::offsets"),
        0);
    ArborX::query(bvh, execspace, predicate, callback, values, offsets);

    Kokkos::View<Coordinates*, ExecSpace> max_radii(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::find_radius::max_radii"),
        this->_clustering_rd_samples);
    Kokkos::parallel_for(
        "RbfPumInterpolator::find_radius::p_for sum max radius",
        Kokkos::RangePolicy(execspace, 0, this->_clustering_rd_samples),
        KOKKOS_LAMBDA(const size_t& i) {
            Coordinates n_max = 0;
            for (size_t ii = offsets(i); ii < offsets(i + 1); ++ii)
            {
                n_max = (n_max > values(ii)) ? n_max : values(ii);
            }
            max_radii(i) = n_max;
        });
    Kokkos::sort(max_radii);
    auto max_radii_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, max_radii);
    this->_radius = std::sqrt(max_radii_h(this->_clustering_rd_samples / 2));
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::find_radius
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
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_lhs"),
        M, N + Pn, N + Pn);
    Kokkos::View<Coordinates**, ExecSpace> all_rhs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_rhs"),
        M, N + Pn);
    auto f = this->_rbf_function;
    auto clusters = Kokkos::create_mirror_view_and_copy(execspace, _clusters);
    auto values = Kokkos::create_mirror_view_and_copy(execspace, _values);
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_lhs rbf "
        "values (A)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU, 0x0LU },
                              { N, N, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            auto rbf_val = f(NDdistance<Dim, Coordinates>(clusters(k, i + 1),
                                                          clusters(k, j + 1)));
            all_lhs(k, i, j) = rbf_val;
            all_lhs(k, j, i) = rbf_val;
            all_rhs(k, i) = values(i);
        });
    auto p = this->_polynomial;
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs poly "
        "values (P/Pt)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU }, { N, M }),
        KOKKOS_LAMBDA(const size_t& j, const size_t& k) {
            auto poly_values = p(clusters(k, 1 + j));
            for (size_t i = 0; i < Pn; ++i)
            {
                all_lhs(k, N + i, j) = poly_values[i];
                all_lhs(k, j, N + i) = poly_values[i];
            }
        });
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs padding "
        "zeros (0)",
        Kokkos::MDRangePolicy(execspace, { N, N, 0x0LU },
                              { N + Pn, N + Pn, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            all_lhs(k, i, j) = 0.0;
            all_rhs(k, i) = 0.0;
        });
    Kokkos::fence();

    // TODO: solve for alpha & beta using KokkosBatched
    const size_t batch_size = M;
    const size_t mat_size = N + Pn;

    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = team_policy::member_type;
    team_policy policy(batch_size, Kokkos::AUTO);

    // const size_t bytes_per_team =
    //     mat_size * (mat_size + 1) * sizeof(Coordinates);
    // policy.set_scratch_size(0, Kokkos::PerTeam(bytes_per_team));

    Kokkos::View<Coordinates**, ExecSpace> coeffs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::prepare_interpolation::coeffs"),
        batch_size, mat_size);

    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for solve systems",
        policy, KOKKOS_LAMBDA(const member_type& member) {
            const size_t i = member.league_rank();

            auto lhs_i =
                Kokkos::subview(all_lhs, i, Kokkos::ALL(), Kokkos::ALL());
            auto rhs_i = Kokkos::subview(all_rhs, i, Kokkos::ALL());

            int r1 = KokkosBatched::TeamTrsm<
                member_type, KokkosBatched::Side::Left,
                KokkosBatched::Uplo::Lower, KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Diag::NonUnit,
                KokkosBatched::Algo::Trsm::Blocked>::invoke(member, 1.0, lhs_i,
                                                            rhs_i);
            member.team_barrier();
            if (r1 != 0)
            {
                Kokkos::printf("Error @ Solve for i=%lu\n", i);
            }

            for (size_t ii = 0; ii < mat_size; ++ii)
            {
                coeffs(i, ii) = rhs_i(ii);
            }
        });

    Kokkos::fence();
}

FULL_TEMPLATE
Coordinates TEMPLATED_CLASSNAME::get_radius() const
{
    return this->_radius;
}

#endif /* ! INTERPOLATOR_HPP */
