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

FULL_TEMPLATE
/* Constructor for the RBF PUM Interpolator:
** @param source: a 1-D `Kokkos::View` which contains source points of the
*field.
** @param values: a 1-D `Kokkos::View` which contains the values associated to
*the source points of the field. Its size is the same as `source`'s.
** @param target: a 1-D `Kokkos::View` which contains the points we want to
*interpolate at.
** @param rbf_function: A user-defined functor which is the rbf function used
*for clusters points evaluation.
*/
template <KokkosViewRank<1> SourceViewType, KokkosViewRank<1> ValuesViewType,
          KokkosViewRank<1> TargetViewType>
TEMPLATED_CLASSNAME::RbfPumInterpolator(
    SourceViewType& source, ValuesViewType& values, TargetViewType& target,
    RbfFunctionBasisType& rbf_function, int nodes_per_cluster,
    double relative_overlap, double rbf_support_radius)
    : _nodes_per_cluster(nodes_per_cluster)
    , _relative_overlap(relative_overlap)
    , _support_radius(rbf_support_radius)
{
    const std::string _region_name = "RbfPumInterpolator::RbfPumInterpolator";
    Kokkos::Profiling::ScopedRegion region(_region_name);

    const ExecSpace execspace{};

    this->_source = decltype(this->_source)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_source"),
        source.extent(0));
    this->_values = decltype(this->_values)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_values"),
        values.extent(0));
    this->_target = decltype(this->_target)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_target"),
        target.extent(0));

    Kokkos::deep_copy(_source, source);
    Kokkos::deep_copy(_values, values);
    Kokkos::deep_copy(_target, target);

    this->_radius = 0.0;
    this->_source_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(this->_source)
    };
    this->_target_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(this->_target)
    };
    this->_rbf_function = rbf_function;
    this->_weighting_function = WendlandC2<RbfPumFPType>{};

    // clang-format off
    // 1. We search for a good cluster radius that would verify:
    // avg(nodes per cluster) ~ this->_nodes_per_clusters
    // To do so, we take the center of the source mesh, and 2 points in each
    // dimension, all at the same distance of the center.
    // We compute from these centers the median max radius to intersect
    // with this->_nodes_per_clusters nodes.
    find_radius();

    this->_rbf_function.set_r_inv(static_cast<RbfPumFPType>(1.0) / this->_support_radius);
    this->_weighting_function.set_r_inv(static_cast<RbfPumFPType>(1.0) / this->_radius);

    create_clusters();

    solve_systems();
    // clang-format on
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
    strs << auto_fp_format<RbfPumFPType>();
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
    strs << "    Execution space: " << ExecSpace{}.name() << "\n";
    strs << "Found radius: " << this->_radius << "\n";
    strs << "Number of clusters: " << this->_clusters.extent(0);

    return strs.str();
}
