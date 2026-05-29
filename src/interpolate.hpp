//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <iostream>
#include <utility>

#include "common/transfer.hxx"
#include "finite_elements/FTInterpClamp.hpp"
#include "finite_elements/FTInterpExtrap.hpp"
#include "finite_elements/FTInterpNearest.hpp"
#include "finite_elements/FTInterpZero.hpp"
#include "finite_elements/FTNearest.hpp"

/**
 * @file interpolate.hpp
 * @brief Generic C++ entry point for PACMAN interpolation methods.
 *
 * This header exposes a single dispatch function that routes a
 * `Transfer<ExecSpace, Dim>` request to the selected PACMAN interpolation
 * backend (finite-elements variants).
 */

namespace PACMAN {
/**
 * @brief Generic C++ interface to PACMAN interpolation methods.
 *
 * The interpolation algorithm is selected through `transfer.method` and then
 * dispatched to the corresponding implementation:
 * - Finite-elements: nearest/nearest, interp-nearest, zero-fill, extrap, clamp
 *
 * @tparam ExecSpace Kokkos execution space used by PACMAN kernels.
 * @tparam Dim Spatial dimension of source/target points.
 * @param[in,out] transfer Transfer descriptor containing source/target data,
 * interpolation method, and output buffers.
 */
template <typename ExecSpace, int_t Dim>
void Interpolate(Transfer<ExecSpace, Dim> &transfer) {
  const TransferMethods method = transfer.method;

  switch (method) {
  case TransferMethods::NEAREST_NEAREST:
    FiniteElements::FTNearest(transfer);
    break;
  case TransferMethods::INTERP_NEAREST:
    FiniteElements::FTInterpNearest(transfer);
    break;
  case TransferMethods::INTERP_ZEROFILL:
    FiniteElements::FTInterZero(transfer);
    break;
  case TransferMethods::INTERP_EXTRAP:
    FiniteElements::FTInterExtrap(transfer);
    break;
  case TransferMethods::INTERP_CLAMP:
    FiniteElements::FTInterClamp(transfer);
    break;
  default:
    std::cerr << "Invalid transfer method" << std::endl;
    assert(false);
    break;
  }
  return;
}
} // namespace PACMAN

// Explicit template instantiation declarations.
// Each enabled exec space has a dedicated ETI translation unit — suppress
// implicit instantiation of Interpolate<ExecSpace, Dim> everywhere else.
#ifndef PACMAN_FE_ETI_COMPILATION
namespace PACMAN {

// clang-format off
#if defined(KOKKOS_ENABLE_SERIAL)
extern template void Interpolate<Kokkos::Serial, 1>(Transfer<Kokkos::Serial, 1> &);
extern template void Interpolate<Kokkos::Serial, 2>(Transfer<Kokkos::Serial, 2> &);
extern template void Interpolate<Kokkos::Serial, 3>(Transfer<Kokkos::Serial, 3> &);
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
extern template void Interpolate<Kokkos::OpenMP, 1>(Transfer<Kokkos::OpenMP, 1> &);
extern template void Interpolate<Kokkos::OpenMP, 2>(Transfer<Kokkos::OpenMP, 2> &);
extern template void Interpolate<Kokkos::OpenMP, 3>(Transfer<Kokkos::OpenMP, 3> &);
#endif

#if defined(KOKKOS_ENABLE_THREADS)
extern template void Interpolate<Kokkos::Threads, 1>(Transfer<Kokkos::Threads, 1> &);
extern template void Interpolate<Kokkos::Threads, 2>(Transfer<Kokkos::Threads, 2> &);
extern template void Interpolate<Kokkos::Threads, 3>(Transfer<Kokkos::Threads, 3> &);
#endif

#if defined(KOKKOS_ENABLE_CUDA)
extern template void Interpolate<Kokkos::Cuda, 1>(Transfer<Kokkos::Cuda, 1> &);
extern template void Interpolate<Kokkos::Cuda, 2>(Transfer<Kokkos::Cuda, 2> &);
extern template void Interpolate<Kokkos::Cuda, 3>(Transfer<Kokkos::Cuda, 3> &);
#endif

#if defined(KOKKOS_ENABLE_HIP)
extern template void Interpolate<Kokkos::HIP, 1>(Transfer<Kokkos::HIP, 1> &);
extern template void Interpolate<Kokkos::HIP, 2>(Transfer<Kokkos::HIP, 2> &);
extern template void Interpolate<Kokkos::HIP, 3>(Transfer<Kokkos::HIP, 3> &);
#endif

#if defined(KOKKOS_ENABLE_SYCL)
extern template void Interpolate<Kokkos::SYCL, 1>(Transfer<Kokkos::SYCL, 1> &);
extern template void Interpolate<Kokkos::SYCL, 2>(Transfer<Kokkos::SYCL, 2> &);
extern template void Interpolate<Kokkos::SYCL, 3>(Transfer<Kokkos::SYCL, 3> &);
#endif
// clang-format on

} // namespace PACMAN
#endif // PACMAN_FE_ETI_COMPILATION
