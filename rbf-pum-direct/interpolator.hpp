#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_LinearBVH.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cache_data.hpp"
#include "callbacks.hpp"
#include "clustering.hpp"
#include "clusters_radius.hpp"
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
    const std::string _region_name = "RbfPumInterpolator::RbfPumInterpolator";
    Kokkos::Profiling::pushRegion(_region_name);
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
    this->_source_bvh =
        ArborX::BoundingVolumeHierarchy{ execspace, this->_source };
    this->_target_bvh =
        ArborX::BoundingVolumeHierarchy{ execspace, this->_target };
    this->_polynomial = polynomial;
    this->_rbf_function = rbf_function;
    this->_weighting_function = WendlandC2<Coordinates>{};
    this->no_data = Point{};
    find_radius();
    this->_rbf_function.set_r_inv(1.0 / this->_radius);
    this->_weighting_function.set_r_inv(1.0 / this->_radius);
    create_clusters();
    prepare_interpolation();
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::RbfPumInterpolator
}

FULL_TEMPLATE
Coordinates TEMPLATED_CLASSNAME::interpolate_at(const Point& target) const
{
    const std::string _region_name = "RbfPumInterpolator::interpolate_at";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};
    size_t N = this->_clusters.extent(0);
    Kokkos::View<Coordinates*, ExecSpace> weights(_region_name + "::weights",
                                                  N);
    Kokkos::View<size_t*, ExecSpace> clusters_index(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::clusters_index"),
        N);
    Kokkos::View<size_t, ExecSpace> id(_region_name + "::id");

    Kokkos::View __clusters =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    Coordinates weights_sum = 0.0;
    auto weighting_function = this->_weighting_function;
    Kokkos::parallel_reduce(
        _region_name
            + "::p_reduce find intersecting "
              "clusters and their weights",
        Kokkos::RangePolicy(execspace, 0, N),
        KOKKOS_LAMBDA(const size_t& i, Coordinates& lsum) {
            const Coordinates w = weighting_function(
                NDdistance<Dim, Coordinates>(target, __clusters(i, 0)));
            if (w > 0)
            {
                const size_t idd = Kokkos::atomic_fetch_inc(&(id()));
                weights(idd) = w;
                lsum += w;
                clusters_index(idd) = i;
            }
        },
        weights_sum);

    if (weights_sum == 0.0)
    {
        std::cerr << "/!\\ Target Point" << point_to_str(target)
                  << "intersects with no cluster" << std::endl;
    }

    size_t M = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, id)();
    Coordinates interpolated_value = 0.0;
    auto rbf_function = this->_rbf_function;
    auto coeffs = Kokkos::create_mirror_view_and_copy(execspace, this->_coeffs);
    auto nb_values_per_cluster = Kokkos::create_mirror_view_and_copy(
        execspace, this->_nb_values_per_cluster);

    Kokkos::parallel_reduce(
        _region_name + "::p_reduce sum local interpolants",
        Kokkos::RangePolicy(execspace, 0, M),
        KOKKOS_LAMBDA(const size_t& i, Coordinates& lsum) {
            // clang-format off
            /* Evaluation formula:
            ** $s(v) = \sum_{k=1}^{N} \lambda_k \, \phi(\|v - x_k\|) + \sum_{m=1}^{M} c_m \, p_m(v)$
            */
            // clang-format on
            for (size_t ii = 0; ii < nb_values_per_cluster(clusters_index(i));
                 ++ii)
            {
                Coordinates rbf = rbf_function(
                    NDdistance(__clusters(clusters_index(i), 1 + ii), target));
                Coordinates local_interpolant =
                    coeffs(clusters_index(i), ii) * rbf;
                lsum += (weights(i) / weights_sum) * local_interpolant;
            }
        },
        interpolated_value);

    Kokkos::Profiling::popRegion();
    return interpolated_value;
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::interpolate(
    PointsView& target, Kokkos::View<Coordinates*, ExecSpace>& out) const
{
    const std::string _region_name = "RbfPumInterpolator::interpolate";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};
    const size_t N = this->_clusters.extent(0);
    const size_t K = target.extent(0);
    Kokkos::View<Coordinates**, ExecSpace> weights(_region_name + "::weights",
                                                   K, N);
    Kokkos::View<bool**, ExecSpace> indices(_region_name + "::indices", K, N);

    auto clusters =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    auto weight_function = this->_weighting_function;

    Kokkos::View<Coordinates*, ExecSpace> weight_sums(
        _region_name + "::weight_sums", K);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    Kokkos::parallel_for(
        _region_name
            + "::p_for compute intersecting clusters and their weights",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& member) {
            const size_t k = member.league_rank();
            Coordinates weight_sum;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, N),
                [&](const size_t& i, Coordinates& lsum) {
                    const Coordinates w =
                        weight_function(NDdistance(target(k), clusters(i, 0)));
                    if (w > 0)
                    {
                        lsum += w;
                        weights(k, i) = w;
                        indices(k, i) = true;
                    }
                },
                weight_sum);
            Kokkos::single(Kokkos::PerTeam(member),
                           [=]() { weight_sums(k) = weight_sum; });
        });
    Kokkos::fence();

    Kokkos::View<Coordinates*, ExecSpace> interpolated_values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::interpolated values"),
        K);

    auto rbf_function = this->_rbf_function;
    auto coeffs = Kokkos::create_mirror_view_and_copy(execspace, this->_coeffs);
    auto bounds = Kokkos::create_mirror_view_and_copy(
        execspace, this->_nb_values_per_cluster);

    Kokkos::parallel_for(
        _region_name + "::p_for build global interpolant",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& member) {
            const size_t k = member.league_rank();
            Coordinates interpolated_value;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, N),
                [&](const size_t& i, Coordinates& lsum) {
                    if (indices(k, i))
                    {
                        for (size_t ii = 0; ii < bounds(i); ++ii)
                        {
                            lsum += (weights(k, i) / weight_sums(k))
                                * (coeffs(i, ii)
                                   * (rbf_function(NDdistance(
                                       clusters(i, 1 + ii), target(k)))));
                        }
                    }
                },
                interpolated_value);
            Kokkos::single(Kokkos::PerTeam(member), [=]() {
                interpolated_values(k) = interpolated_value;
            });
        });

    Kokkos::fence();

    out = decltype(interpolated_values)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::out"),
        interpolated_values.extent(0));
    Kokkos::deep_copy(out, interpolated_values);

    Kokkos::Profiling::popRegion(); // ! Interpolate
}

FULL_TEMPLATE
std::string TEMPLATED_CLASSNAME::get_interpolator_details(void) const
{
    std::ostringstream strs;
    strs << "#Source points: " << this->_source.extent(0) << "\n";
    strs << "#Values: " << this->_values.extent(0) << "\n";
    strs << "#Target points: " << this->_target.extent(0) << "\n";
    strs << "Source bounding box:\n";
    strs << "    Lower: "
         << point_to_str(this->_source_bvh.bounds().minCorner()) << "\n";
    strs << "    Upper: "
         << point_to_str(this->_source_bvh.bounds().maxCorner()) << "\n";
    strs << "Interpolation params:\n";
    strs << "    #Points per cluster: " << this->_nodes_per_cluster << "\n";
    strs << "    Relative overlap: " << this->_relative_overlap << "\n";
    strs << "    Project to input: " << std::boolalpha
         << this->_project_to_input << "\n";
    strs << "    RBF Function: " << typeid(RbfFunctionBasisType).name() << "\n";
    strs << "    Polynomial Function: " << typeid(PolynomialType).name()
         << "\n";
    strs << "Found radius: " << std::setprecision(17) << this->_radius << "\n";
    strs << get_clusters_info(_nb_values_per_cluster);

    return strs.str();
}

#endif /* ! INTERPOLATOR_HPP */
