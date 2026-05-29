//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

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

// Explicit template instantiation declarations.
// Each enabled exec space has a dedicated ETI translation unit that provides
// the definitions — suppress implicit instantiation everywhere else.
#ifndef PACMAN_RBF_ETI_COMPILATION
namespace PACMAN {
namespace RbfPum {

// clang-format off
#if defined(KOKKOS_ENABLE_SERIAL)
extern template class RbfPumInterpolator<Kokkos::Serial, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Serial, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Serial, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Serial, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Serial, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Serial, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Serial, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Serial, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Serial, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Serial, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Serial, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Serial, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Serial, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Serial, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Serial, 3, WendlandC8>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
extern template class RbfPumInterpolator<Kokkos::OpenMP, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::OpenMP, 3, WendlandC8>;
#endif

#if defined(KOKKOS_ENABLE_THREADS)
extern template class RbfPumInterpolator<Kokkos::Threads, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Threads, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Threads, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Threads, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Threads, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Threads, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Threads, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Threads, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Threads, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Threads, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Threads, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Threads, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Threads, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Threads, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Threads, 3, WendlandC8>;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
extern template class RbfPumInterpolator<Kokkos::Cuda, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::Cuda, 3, WendlandC8>;
#endif

#if defined(KOKKOS_ENABLE_HIP)
extern template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC8>;
#endif

#if defined(KOKKOS_ENABLE_SYCL)
extern template class RbfPumInterpolator<Kokkos::SYCL, 1, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 1, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 1, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 1, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 1, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 2, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 2, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 2, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 2, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 2, WendlandC8>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 3, WendlandC0>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 3, WendlandC2>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 3, WendlandC4>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 3, WendlandC6>;
extern template class RbfPumInterpolator<Kokkos::SYCL, 3, WendlandC8>;
#endif
// clang-format on

} // namespace RbfPum
} // namespace PACMAN
#endif // PACMAN_RBF_ETI_COMPILATION
