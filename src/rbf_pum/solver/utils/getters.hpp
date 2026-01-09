#pragma once

#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "utils/utils.hpp"

namespace PACMAN {
namespace RbfPum {

template <KokkosViewRank<1> AsViewType, KokkosViewRank<1> OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfMatrix(const index_t &k, const AsViewType &As, const OffsViewType &offs,
             const int_t &n, const int_t &m) {
  using DataType = typename AsViewType::non_const_value_type;
  using MemorySpace = typename AsViewType::memory_space;
  const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
  auto A_data = Kokkos::subview(As, slice);
  return Kokkos::View<DataType **, MemorySpace,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(A_data.data(), n,
                                                               m);
}

template <KokkosViewRank<1> PsViewType, KokkosViewRank<1> OffsViewType,
          KokkosArray AxisType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetPolyMatrix(const index_t &k, const PsViewType &Ps, const OffsViewType &offs,
              const int_t &n, const AxisType &activeAxis) {
  using DataType = typename PsViewType::non_const_value_type;
  using MemorySpace = typename PsViewType::memory_space;
  const int_t N = offs(k + 1) - offs(k);
  int_t M = activeAxis.size() + 1;
  const auto slice = Kokkos::make_pair(offs(k) * M, offs(k + 1) * M);
  auto P_data = Kokkos::subview(Ps, slice);
  M = 1;
  for (int_t d = 0; d < activeAxis.size(); ++d) {
    if (activeAxis[d]) {
      ++M;
    }
  }

  return Kokkos::View<DataType **, MemorySpace,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(P_data.data(), n,
                                                               M);
}

template <KokkosViewRank<1> BsViewType, KokkosViewRank<1> OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfVector(const index_t &k, const BsViewType &Bs, const OffsViewType &offs,
             const int_t &n) {
  using DataType = typename BsViewType::non_const_value_type;
  using MemorySpace = typename BsViewType::memory_space;
  const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
  auto B_data = Kokkos::subview(Bs, slice);
  return Kokkos::View<DataType *, MemorySpace,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(B_data.data(),
                                                               n);
}

} // namespace RbfPum
} // namespace PACMAN
