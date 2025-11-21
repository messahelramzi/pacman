#pragma once

#include <ArborX_LinearBVH.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "callbacks.hpp"
#include "clustering/clustering.hpp"
#include "clustering/clusters_radius.hpp"
#include "interpolator.hxx"
#include "solver/rbf_functions.hpp"
#include "solver/solver.hpp"
#include "utils/operators.hpp"
#include "utils/utils.hpp"

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
TEMPLATED_CLASSNAME::RbfPumInterpolator(VectorView<Point> source,
                                        VectorView<ScalarType> values,
                                        VectorView<Point> target,
                                        RbfFunctionBasisType rbf_function)
{
    const std::string _region_name = "RbfPumInterpolator::RbfPumInterpolator";
    Kokkos::Profiling::pushRegion(_region_name);

    ExecSpace execspace{};

    this->_source = decltype(this->_source)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_source"),
        source.extent(0));
    Kokkos::deep_copy(_source, source);

    this->_values = decltype(this->_values)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_values"),
        values.extent(0));
    Kokkos::deep_copy(_values, values);

    this->_target = decltype(this->_target)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_target"),
        target.extent(0));
    Kokkos::deep_copy(_target, target);

    this->_radius = 0.0;
    this->_source_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(this->_source)
    };
    this->_target_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(this->_target)
    };
    this->_rbf_function = rbf_function;
    this->_weighting_function = WendlandC2<ScalarType>{};

    // clang-format off
    // 1. We search for a good cluster radius that would verify:
    // avg(nodes per cluster) ~ this->_nodes_per_clusters
    // To do so, we take the center of the source mesh, and 2 points in each
    // dimension, all at the same distance of the center.
    // We compute from these centers the median max radius to intersect
    // with this->_nodes_per_clusters nodes.
    find_radius();

    this->_rbf_function.set_r_inv(1.0 / this->_support_radius);
    this->_weighting_function.set_r_inv(1.0 / this->_support_radius);

    create_clusters();

    solve_systems();

    // clang-format on
    Kokkos::Profiling::popRegion(); // ! RbfPumInterpolator::RbfPumInterpolator
}

FULL_TEMPLATE
/* Interpolates a value for each point of `target` and returns the values in
 * `out`.
 * @param target: a 1xN Kokkos::View of Points.
 * @param out: output parameter, a 1xN Kokkos::View containing the interpolated
 * values.
 * @note `out` doesn't need to be pre-allocated as `interpolate` overrides it.
 */
void TEMPLATED_CLASSNAME::interpolate(VectorView<Point>& target,
                                      VectorView<ScalarType>& out) const
{
    assert(this->_radius > 0);
    assert(this->_clusters.extent(0) > 0);
    assert(this->_coeffs.size() > 0);
    const std::string _region_name = "RbfPumInterpolator::interpolate";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};
    const size_t N = target.extent(0);

    // 1. On crée une BVH sur les centres des clusters pour pouvoir trouver
    // efficacement l'appartenance aux clusters sur les points target
    const VectorView<Point> centers =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    auto centers_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(centers)
    };

    // 2. Pour chaque point target, on récupère les clusters auxquels il
    // appartient
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_clusters_predicate(
        target, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType> get_clusters_callback;

    VectorView<unsigned int> targets_clusters(
        _region_name + "::targets_clusters", 0);
    VectorView<int> targets_clusters_offsets(
        _region_name + "::targets_clusters_offsets", 0);

    centers_bvh.query(execspace, get_clusters_predicate, get_clusters_callback,
                      targets_clusters, targets_clusters_offsets);

    // 3. Pour chaque cluster de chaque point target, on calcule:
    // ɸ(norm2(x, x_j)), avec ɸ = this->_weighting_function
    VectorView<ScalarType> weights(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::weights"),
        targets_clusters.extent(0));

    // fonction WendlandC2
    auto weighting_function = this->_weighting_function;

    // fonction rbf
    auto rbf_function = this->_rbf_function;

    Kokkos::parallel_for(
        _region_name + "::p_for compute weights",
        Kokkos::RangePolicy(execspace, 0, N), KOKKOS_LAMBDA(const size_t& i) {
            ScalarType lsum = 0.0;
            for (int k = targets_clusters_offsets(i);
                 k < targets_clusters_offsets(i + 1); ++k)
            {
                const auto center = centers(targets_clusters(k));
                ScalarType w =
                    weighting_function.eval(NDdistance(target(i), center));
                weights(k) = w;
                lsum += w;
            }
            // 4. On normalise les poids pour que leur somme fasse 1
            for (int k = targets_clusters_offsets(i);
                 k < targets_clusters_offsets(i + 1); ++k)
            {
                weights(k) /= lsum;
            }
        });
    Kokkos::fence();

    Kokkos::realloc(out, target.extent(0));
}

FULL_TEMPLATE
/* @return: a string that contains information about the internal state
 * of the interpolator.
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
    strs << "    Lower: " << this->_source_bvh.bounds().minCorner() << "\n";
    strs << "    Upper: " << this->_source_bvh.bounds().maxCorner() << "\n";
    strs << "Interpolation params:\n";
    strs << "    #Points per cluster: " << this->_nodes_per_cluster << "\n";
    strs << "    Relative overlap: " << this->_relative_overlap << "\n";
    strs << "    RBF Function: " << typeid(RbfFunctionBasisType).name() << "\n";
    strs << "Found radius: " << std::setprecision(16) << this->_radius << "\n";
    strs << "Number of clusters: " << this->_clusters.extent(0);

    return strs.str();
}
