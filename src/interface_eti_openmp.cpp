//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for PACMAN::Interpolate — Kokkos::OpenMP.
#define PACMAN_FE_ETI_COMPILATION
#include "interpolate.hpp"

#if defined(KOKKOS_ENABLE_OPENMP)
namespace PACMAN {
// clang-format off
template void Interpolate<Kokkos::OpenMP, 1>(Transfer<Kokkos::OpenMP, 1> &);
template void Interpolate<Kokkos::OpenMP, 2>(Transfer<Kokkos::OpenMP, 2> &);
template void Interpolate<Kokkos::OpenMP, 3>(Transfer<Kokkos::OpenMP, 3> &);
// clang-format on
} // namespace PACMAN
#endif
