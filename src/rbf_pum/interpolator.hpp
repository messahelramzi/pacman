//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

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

/// @brief Builds an interpolator object from the `Transfer` argument, and fills
/// `rTransfer.targetValues` with the interpolated values.
/// @tparam ExecSpace the Kokkos execution space interpolation will happen in
/// @tparam Dim space dimension (1 <= Dim <= 3)
/// @tparam RbfFunctionBasisType the RBF function to use for the RBF-PUM problem
/// @param[in,out] rTransfer The `Transfer` object, which holds the field
/// transfer information
/// @param[in] nodesPerCluster An average number of nodes per cluster, defaults
/// to `50`
/// @param[in] relativeOverlap  The minimum overlap ratio in [0,1[ used for
/// clustering, defaults to `0.15`
/// @param[in] rbfSupportRadius The support radius of the RBF function passed as
/// a template argument, defaults to `0.1`
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

  FindRadius();
  this->mRbfFunction.SetRadiusInv(fp_consts::one() / this->mSupportRadius);
  this->mWeightingFunction.SetRadiusInv(fp_consts::one() / this->mRadius);
  CreateClusters();
  Interpolate();
}

/// @brief builds a string that contains information about the internal state
///        of the interpolator.
/// @tparam ExecSpace the Kokkos execution space interpolation will happen in
/// @tparam Dim space dimension (1 <= Dim <= 3)
/// @tparam RbfFunctionBasisType the RBF function to use for the RBF-PUM problem
/// @return string
/// @warning this function must not be called before the full initialization
///          of the clusters of the interpolator object.
FULL_TEMPLATE
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
