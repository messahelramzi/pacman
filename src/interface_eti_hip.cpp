//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::HIP.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_HIP)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::HIP, 1>(Transfer<Kokkos::HIP, 1> &);
template void Interpolate<Kokkos::HIP, 2>(Transfer<Kokkos::HIP, 2> &);
template void Interpolate<Kokkos::HIP, 3>(Transfer<Kokkos::HIP, 3> &);
// clang-format on
} // namespace PACMAN
#endif
