//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::Serial.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_SERIAL)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::Serial, 1>(Transfer<Kokkos::Serial, 1> &);
template void Interpolate<Kokkos::Serial, 2>(Transfer<Kokkos::Serial, 2> &);
template void Interpolate<Kokkos::Serial, 3>(Transfer<Kokkos::Serial, 3> &);
// clang-format on
} // namespace PACMAN
#endif
