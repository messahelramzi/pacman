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
#include "common/transfer.hxx"
#include "common/types.hpp"
#include "interpolator.hxx"
#include "solver/rbf_functions.hpp"
#include "solver/solver.hpp"
#include "utils/operators.hpp"
#include "utils/utils.hpp"

namespace PACMAN {
namespace RbfPum {
template <KokkosViewRank<2> SourceType, KokkosViewRank<1> TargetType, int_t Dim>
static inline void FillTarget(const SourceType &sourceView,
                              TargetType &targetView) {
  assert(Dim == sourceView.extent_int(1));
  using ExecSpace = typename SourceType::execution_space;
  const int K = sourceView.extent_int(0);
  Kokkos::parallel_for(
      "FillTarget", Kokkos::RangePolicy(ExecSpace{}, 0, K),
      KOKKOS_LAMBDA(const int &k) {
        ::ArborX::Point<Dim, coordinates_t> point{};
        for (int_t axis = 0; axis < Dim; ++axis) {
          point[axis] = sourceView(k, axis);
        }
        targetView(k) = point;
      });
}

FULL_TEMPLATE
TEMPLATED_CLASSNAME::RbfPumInterpolator(Transfer<ExecSpace, Dim> &rTransfer,
                                        int_t nodesPerCluster,
                                        fp_t relativeOverlap,
                                        fp_t rbfSupportRadius)
    : mNodesPerCluster(nodesPerCluster), mRelativeOverlap(relativeOverlap),
      mSupportRadius(rbfSupportRadius) {
  const std::string _region_name = "RbfPumInterpolator::RbfPumInterpolator";
  Kokkos::Profiling::ScopedRegion region(_region_name);

  const ExecSpace execspace{};

  this->mSource = rTransfer.sourcePoints;
  this->mValues = rTransfer.sourceValues;
  this->mTarget = VectorView<Point>(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "this->mTarget"),
      rTransfer.targetPoints.extent(0));
  FillTarget<decltype(rTransfer.targetPoints), decltype(this->mTarget), Dim>(
      rTransfer.targetPoints, this->mTarget);
  this->out = rTransfer.targetValues;

  this->mRadius = fp_consts::zero();
  this->mSourceBvh = ::ArborX::BoundingVolumeHierarchy{
      execspace, ::ArborX::Experimental::attach_indices(this->mSource)};
  this->mTargetBvh = ::ArborX::BoundingVolumeHierarchy{
      execspace, ::ArborX::Experimental::attach_indices(this->mTarget)};
  this->mRbfFunction = RbfFunctionBasisType{};
  this->mWeightingFunction = WendlandC2{};
  Kokkos::fence();

  // 1. We search for a good cluster radius that would verify:
  // avg(nodes per cluster) ~ this->_nodes_per_clusters
  // To do so, we take the center of the source mesh, and 2 points in each
  // dimension, all at the same distance of the center.
  // We compute from these centers the median max radius to intersect
  // with this->_nodes_per_clusters nodes.
  FindRadius();

  this->mRbfFunction.SetRadiusInv(fp_consts::one() / this->mSupportRadius);
  this->mWeightingFunction.SetRadiusInv(fp_consts::one() / this->mRadius);

  CreateClusters();

  Interpolate();
}

FULL_TEMPLATE
/* @return: a string that contains information about the internal state
 * of the interpolator.
 * @warning this function must not be called before the full initialization
 * of the clusters of the interpolator object.
 */
std::string TEMPLATED_CLASSNAME::GetInterpolatorDetails(void) const {
  std::ostringstream strs;
  strs << fp_consts::set_precision();
  strs << "#Source points: " << this->mSource.extent(0) << "\n";
  strs << "#Values: " << this->mValues.extent(0) << "\n";
  strs << "#Target points: " << this->mTarget.extent(0) << "\n";
  strs << "Source bounding box:\n";
  strs << "    Lower: " << this->mSourceBvh.bounds().minCorner() << "\n";
  strs << "    Upper: " << this->mSourceBvh.bounds().maxCorner() << "\n";
  strs << "Interpolation params:\n";
  strs << "    #Points per cluster: " << this->mNodesPerCluster << "\n";
  strs << "    Relative overlap: " << this->mRelativeOverlap << "\n";
  strs << "    RBF Function: " << typeid(RbfFunctionBasisType).name() << "\n";
  strs << "    Execution space: " << ExecSpace{}.name() << "\n";
  strs << "Found radius: " << this->mRadius << "\n";
  strs << "Number of clusters: " << this->mClusters.extent(0);

  return strs.str();
}

} // namespace RbfPum
} // namespace PACMAN
