//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::Cuda.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::Cuda, 1>(Transfer<Kokkos::Cuda, 1> &);
template void Interpolate<Kokkos::Cuda, 2>(Transfer<Kokkos::Cuda, 2> &);
template void Interpolate<Kokkos::Cuda, 3>(Transfer<Kokkos::Cuda, 3> &);
// clang-format on
} // namespace PACMAN
#endif
