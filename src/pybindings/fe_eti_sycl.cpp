//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::SYCL.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_SYCL)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::SYCL, 1>(Transfer<Kokkos::SYCL, 1> &);
template void Interpolate<Kokkos::SYCL, 2>(Transfer<Kokkos::SYCL, 2> &);
template void Interpolate<Kokkos::SYCL, 3>(Transfer<Kokkos::SYCL, 3> &);
// clang-format on
} // namespace PACMAN
#endif
