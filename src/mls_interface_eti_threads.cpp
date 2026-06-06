//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for MLSInterpolator — Kokkos::Threads.
#define PACMAN_MLS_ETI_COMPILATION
#include "mls/interpolator.hpp"

#if defined(KOKKOS_ENABLE_THREADS)
namespace PACMAN {
namespace MLS {
// clang-format off
template class MLSInterpolator<Kokkos::Threads, 1>;
template class MLSInterpolator<Kokkos::Threads, 2>;
template class MLSInterpolator<Kokkos::Threads, 3>;
// clang-format on
} // namespace MLS
} // namespace PACMAN
#endif
