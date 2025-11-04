#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <ArborX_LinearBVH.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "batched_operations.hpp"
#include "cache_data.hpp"
#include "callbacks.hpp"
#include "clustering.hpp"
#include "clusters_radius.hpp"
#include "interpolator.hxx"
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
** @param rbf_function: A user-defined functor which is the rbf function used
*for clusters points evaluation.
*/
FULL_TEMPLATE
TEMPLATED_CLASSNAME::RbfPumInterpolator(
    PointsView source, Kokkos::View<Coordinates*, ExecSpace> values,
    PointsView target, RbfFunctionBasisType rbf_function)
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
    this->_rbf_function = rbf_function;
    this->_weighting_function = WendlandC2<Coordinates>{};

    // clang-format off
    // 1. We search for a good cluster radius that would verify:
    // avg(nodes per cluster) ~ this->_nodes_per_clusters
    // To do so, we take the center of the source mesh, and 2 points in each
    // dimension, all at the same distance of the center.
    // We compute from these centers the median max radius to intersect
    // with this->_nodes_per_clusters nodes.
    find_radius();

    // 2. We set the support radius of our RBF functions such as:
    // support radius = cluster radius
    // Thus, a point is in the support radius of a cluster if:
    // norm2(point, cluster center) * _r_inv < 1
    // Here, the clusters rbf function is user defined, but
    // the weighting function is fixed (Wendland C2).
    this->_rbf_function.set_r_inv(1.0 / this->_radius);
    this->_weighting_function.set_r_inv(1.0 / this->_radius);

    // 3. We create clusters following this algorithm:
    //      -> We compute the number of centers that we want to try as clusters centers
    //      -> Then we keep only the clusters which intersect with the source mesh
    //      -> Then, we keep only the clusters which intersect with the target mesh
    //      -> Then, we remove the clusters centers which verify:
    //             norm2(center1, center2) < 0.4 * ~radius
    //      -> Then, we keep project the centers on the source mesh if this->_project_to_input is true
    //      -> We remove one more time the centers that are not intersecting with the target mesh
    //      -> Finally, we create the clusters, in this->_clusters, a MxN+1 points matrix:
    //          this->_clusters = [[center1, value1, value2, value3, ..., valuen],
    //                             [center2, value1, value2, value3, ..., valuen],
    //                             [...      ...     ...     ...     ...  ...   ],
    //                             [centerm, value1, value2, value3, ..., valuen]]
    //   As some clusters contain less values than others, the keep track of the effective
    //   values of each clusters in this->_bounds.
    create_clusters();

    // 4. We compute the coefficients α and β of each cluster by solving:
    //
    // | A   P | | α | = | u |
    // | P^T 0 | | β |   | 0 |
    //
    // With:
    // A the RBF coefficients matrix
    // P the linear polynomial augmentation matrix
    // α the coefficients for the RBF part of the local interpolant
    // β the coefficients for the polynomial part of the local interpolant
    // u the values associated of the source nodes
    prepare_interpolation();

    // clang-format on
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::RbfPumInterpolator
}

FULL_TEMPLATE
/* Interpolates value for a single point.
 * @param target: the target point for interpolation.
 * @warning This method should not be used when we know in advance all
 * the points we interpolate on. Use interpolate instead.
 */
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

#define MIN(A, B) ((A) < (B)) ? (A) : (B)

FULL_TEMPLATE
/* Interpolates a value for each point of `target` and returns the values in
 * `out`.
 * @param target: a 1xN Kokkos::View of Points.
 * @param out: output parameter, a 1xN Kokkos::View containing the interpolated
 * values.
 * @note `out` doesn't need to be pre-allocated as `interpolate` overrides it.
 */
void TEMPLATED_CLASSNAME::interpolate(
    PointsView& target, Kokkos::View<Coordinates*, ExecSpace>& out) const
{
    const std::string _region_name = "RbfPumInterpolator::interpolate";
    Kokkos::Profiling::pushRegion(_region_name);
    const size_t BATCH_SIZE = 16384;
    size_t index = 0;
    const size_t upper_bound = target.extent(0);

    Kokkos::realloc(out, upper_bound);

    while (index < upper_bound)
    {
        auto index_range =
            Kokkos::make_pair(index, MIN(index + BATCH_SIZE, upper_bound));
        auto batch = Kokkos::subview(target, index_range);
        auto batch_out = Kokkos::subview(out, index_range);
        batched_interpolate(batch, batch_out);
        index = index_range.second;
    }
    Kokkos::Profiling::popRegion(); // ! interpolate
}

FULL_TEMPLATE
/* @return: a string that contains information about the internal state
 * of the allocator.
 * @warning this function must not be called before the full initialization
 * of the clusters of the interpolator object.
 */
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
    strs << "Found radius: " << std::setprecision(17) << this->_radius << "\n";
    strs << get_clusters_info(_nb_values_per_cluster);

    return strs.str();
}

#endif /* ! INTERPOLATOR_HPP */
