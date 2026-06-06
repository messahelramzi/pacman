//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX_InterpMovingLeastSquares.hpp>
#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"

#define PACMAN_MLS_FULL_TEMPLATE template <typename ExecSpace, int_t Dim>
#define PACMAN_MLS_CLASSNAME MLSInterpolator<ExecSpace, Dim>

namespace PACMAN {
namespace MLS {

PACMAN_MLS_FULL_TEMPLATE
class MLSInterpolator {
  using Point = ::ArborX::Point<Dim, coordinates_t>;
  using MemSpace = typename ExecSpace::memory_space;

  template <typename inner_type>
  using VectorView = Kokkos::View<inner_type *, MemSpace>;

public:
  MLSInterpolator(Transfer<ExecSpace, Dim> &rTransfer);

  VectorView<fp_t> out;
};

} // namespace MLS
} // namespace PACMAN

// Explicit template instantiation declarations.
// Each enabled exec space has a dedicated ETI translation unit that provides
// the definitions — suppress implicit instantiation everywhere else.
#ifndef PACMAN_MLS_ETI_COMPILATION
namespace PACMAN {
namespace MLS {

// clang-format off
#if defined(KOKKOS_ENABLE_SERIAL)
extern template class MLSInterpolator<Kokkos::Serial, 1>;
extern template class MLSInterpolator<Kokkos::Serial, 2>;
extern template class MLSInterpolator<Kokkos::Serial, 3>;
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
extern template class MLSInterpolator<Kokkos::OpenMP, 1>;
extern template class MLSInterpolator<Kokkos::OpenMP, 2>;
extern template class MLSInterpolator<Kokkos::OpenMP, 3>;
#endif

#if defined(KOKKOS_ENABLE_THREADS)
extern template class MLSInterpolator<Kokkos::Threads, 1>;
extern template class MLSInterpolator<Kokkos::Threads, 2>;
extern template class MLSInterpolator<Kokkos::Threads, 3>;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
extern template class MLSInterpolator<Kokkos::Cuda, 1>;
extern template class MLSInterpolator<Kokkos::Cuda, 2>;
extern template class MLSInterpolator<Kokkos::Cuda, 3>;
#endif

#if defined(KOKKOS_ENABLE_HIP)
extern template class MLSInterpolator<Kokkos::HIP, 1>;
extern template class MLSInterpolator<Kokkos::HIP, 2>;
extern template class MLSInterpolator<Kokkos::HIP, 3>;
#endif

#if defined(KOKKOS_ENABLE_SYCL)
extern template class MLSInterpolator<Kokkos::SYCL, 1>;
extern template class MLSInterpolator<Kokkos::SYCL, 2>;
extern template class MLSInterpolator<Kokkos::SYCL, 3>;
#endif
// clang-format on

} // namespace MLS
} // namespace PACMAN
#endif // PACMAN_MLS_ETI_COMPILATION
