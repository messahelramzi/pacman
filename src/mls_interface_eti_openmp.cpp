//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for MLSInterpolator — Kokkos::OpenMP.
#define PACMAN_MLS_ETI_COMPILATION
#include "mls/interpolator.hpp"

#if defined(KOKKOS_ENABLE_OPENMP)
namespace PACMAN {
namespace MLS {
// clang-format off
template class MLSInterpolator<Kokkos::OpenMP, 1>;
template class MLSInterpolator<Kokkos::OpenMP, 2>;
template class MLSInterpolator<Kokkos::OpenMP, 3>;
// clang-format on
} // namespace MLS
} // namespace PACMAN
#endif
