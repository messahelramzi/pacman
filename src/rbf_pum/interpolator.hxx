#pragma once

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"
#include "solver/rbf_functions.hpp"

#define FULL_TEMPLATE                                                          \
  template <typename ExecSpace, int_t Dim, typename RbfFunctionBasisType>

#define TEMPLATED_CLASSNAME                                                    \
  RbfPumInterpolator<ExecSpace, Dim, RbfFunctionBasisType>

namespace PACMAN {
namespace RbfPum {
FULL_TEMPLATE
class RbfPumInterpolator {
  using Point = ::ArborX::Point<Dim, coordinates_t>;
  using MemSpace = typename ExecSpace::memory_space;

  template <typename inner_type>
  using VectorView = Kokkos::View<inner_type *, MemSpace>;

  template <typename inner_type>
  using MatrixView = Kokkos::View<inner_type **, MemSpace>;

public:
  // Constructor
  RbfPumInterpolator(Transfer<ExecSpace, Dim> &rTransfer,
                     int_t nodesPerCluster = 50, fp_t relativeOverlap = 0.15,
                     fp_t rbfSupportRadius = 0.1);

  // Internal routines
  void FindRadius(void);
  void CreateClusters(void);
  void Interpolate(void);

  // Debug routines
  std::string GetInterpolatorDetails(void) const;

  VectorView<fp_t> out;

private:
  coordinates_t mRadius;
  int_t mNodesPerCluster;
  fp_t mRelativeOverlap;
  fp_t mSupportRadius;

  ::ArborX::BoundingVolumeHierarchy<typename VectorView<Point>::memory_space,
                                    ::ArborX::PairValueIndex<Point, index_t>>
      mSourceBvh;
  ::ArborX::BoundingVolumeHierarchy<typename VectorView<Point>::memory_space,
                                    ::ArborX::PairValueIndex<Point, index_t>>
      mTargetBvh;
  RbfFunctionBasisType mRbfFunction;
  VectorView<Point> mClusters;
  VectorView<Point> mSource;
  VectorView<Point> mTarget;
  VectorView<fp_t> mCoeffs;
  VectorView<fp_t> mValues;
  WendlandC2 mWeightingFunction;
};

} // namespace RbfPum
} // namespace PACMAN
