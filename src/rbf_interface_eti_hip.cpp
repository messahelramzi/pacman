//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//
// Explicit template instantiation for RbfPumInterpolator — Kokkos::HIP.
#define PACMAN_RBF_ETI_COMPILATION
#include "rbf_pum/interpolator.hpp"

#if defined(KOKKOS_ENABLE_HIP)
namespace PACMAN {
namespace RbfPum {
// clang-format off
template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC0>;
template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC2>;
template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC4>;
template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC6>;
template class RbfPumInterpolator<Kokkos::HIP, 1, WendlandC8>;
template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC0>;
template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC2>;
template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC4>;
template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC6>;
template class RbfPumInterpolator<Kokkos::HIP, 2, WendlandC8>;
template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC0>;
template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC2>;
template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC4>;
template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC6>;
template class RbfPumInterpolator<Kokkos::HIP, 3, WendlandC8>;
// clang-format on
} // namespace RbfPum
} // namespace PACMAN
#endif
