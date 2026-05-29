//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "utils/utils.hpp"

namespace PACMAN {
namespace RbfPum {

/// @brief Returns a matrix of rank 2, shaped as (n, m), using the given data
/// view, team rank, and offsets. This function basically reinterprets a 1D
/// vector into a 2D matrix accordingly to the given dimensions.
/// @tparam AsViewType Matrices data view type.
/// @tparam OffsViewType Access offsets type for the given data view.
/// @param k The index of the current team in the league
/// @param As The matrices data view
/// @param offs The access offsets of the matrices data view
/// @param n The number of lines in the reshaped matrix which is returned
/// @param m The number of columns in the reshaped matrix which is returned
/// @return A non owning data `Kokkos::View` (unmanaged) which represents the
/// RBF matrix.
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

/// @brief Returns a matrix of rank 2, shaped as (n, m), using the given data
/// view, team rank, and offsets. This function basically reinterprets a 1D
/// vector into a 2D matrix accordingly to the given dimension and active axis.
/// @tparam PsViewType Matrices data view type.
/// @tparam OffsViewType Access offsets type for the given data view.
/// @tparam AxisType Active axis `Kokkos::Array` type
/// @param k The index of the current team in the league
/// @param Ps The matrices data view
/// @param offs The access offsets of the matrices data view
/// @param n The number of lines in the reshaped matrix which is returned
/// @param activeAxis The number of active axis is the number of columns in the
/// reshaped matrix which is returned
/// @return A non owning data `Kokkos::View` (unmanaged) which represents the
/// polynomial matrix.
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

/// @brief Returns a vector view of rank 1, of length n, using the given data
/// view, team rank, and offsets. This function wraps a slice of `Bs` into an
/// unmanaged view.
/// @tparam BsViewType Vectors data view type.
/// @tparam OffsViewType Access offsets type for the given data view.
/// @param k The index of the current team in the league
/// @param Bs The vectors data view
/// @param offs The access offsets of the vectors data view
/// @param n The length of the returned vector
/// @return A non owning data `Kokkos::View` (unmanaged) which represents the
/// source values vector.
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
