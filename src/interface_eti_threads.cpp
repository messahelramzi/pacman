//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::Threads.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_THREADS)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::Threads, 1>(Transfer<Kokkos::Threads, 1> &);
template void Interpolate<Kokkos::Threads, 2>(Transfer<Kokkos::Threads, 2> &);
template void Interpolate<Kokkos::Threads, 3>(Transfer<Kokkos::Threads, 3> &);
// clang-format on
} // namespace PACMAN
#endif
