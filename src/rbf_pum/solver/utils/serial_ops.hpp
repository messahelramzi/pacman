//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "solver/utils/serial_solve.hpp"

namespace PACMAN {
namespace RbfPum {

/// @brief Deactivate one axis, the one with the smallest range of values and
/// returns the index of this deactivated axis
/// @tparam SourceType Source points view type
/// @tparam OffsType Access offsets view
/// @tparam AxisType A `Kokkos::Array<Dim>`
/// @param src Source points view
/// @param offs Access offsets for the source points views, to access the points
/// of this local cluster
/// @param activeAxis A `Kokkos::Array` which holds the axis status
/// @return The index of the deactivated axis
template <KokkosViewRank<1> SourceType, KokkosViewRank<1> OffsType,
          KokkosArray AxisType, int_t dim = AxisType::size()>
KOKKOS_INLINE_FUNCTION int_t SerialDeactivateOneAxis(const SourceType &src,
                                                     const OffsType &offs,
                                                     AxisType &activeAxis) {
  using ExecSpace = typename base_type<SourceType>::execution_space;
  Kokkos::Array<fp_t, dim> distances;
  for (int_t axis = 0; axis < dim; ++axis) {
    if (!activeAxis[axis]) {
      distances[axis] = fp_consts::max();
      continue;
    }
    auto min_max_pair = RbfPumMinMax(offs, [=](const int &i, const int &j) {
      return src(i)[axis] < src(j)[axis];
    });
    const fp_t dist = Kokkos::abs(src(min_max_pair.second)[axis] -
                                  src(min_max_pair.first)[axis]);
    distances[axis] = dist;
  }
  fp_t min = fp_consts::max();
  int_t min_i = 0;
  for (int_t axis = 0; axis < dim; ++axis) {
    if (distances[axis] < min) {
      min = distances[axis];
      min_i = axis;
    }
  }
  activeAxis[min_i] = false;
  return min_i;
}

/// @brief Fill the polynomial augmentation matrix accordingly to the activated
/// axis in `activeAxis`
/// @tparam PViewType The polynomial matrix type
/// @tparam XViewType Source point view type
/// @tparam OffsViewType Access offsets for the source view type
/// @tparam AxisType A `Kokkos::Array<Dim>`
/// @param P Polynomial augmentation matrix (Q or V)
/// @param X Source points view
/// @param XOffs Access offsets for the source points views, to access the
/// points of this local cluster
/// @param activeAxis A `Kokkos::Array` which holds the axis status
template <KokkosViewRank<2> PViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> OffsViewType, KokkosArray AxisType,
          int_t dim = AxisType::size()>
KOKKOS_FUNCTION void SerialFillPoly(PViewType &P, const XViewType &X,
                                    const OffsViewType &XOffs,
                                    AxisType &activeAxis) {
  const int_t N = P.extent_int(0);
  int_t active_count = 0;
  int_t active_index[dim];
  for (int_t j = 0; j < dim; ++j) {
    if (activeAxis[j]) {
      active_index[active_count++] = j;
    }
  }
  for (int_t i = 0; i < N; ++i) {
    P(i, 0) = fp_consts::one();
    const auto x = X(XOffs(i));
    for (int_t k = 0; k < active_count; ++k) {
      const int_t j = active_index[k];
      P(i, 1 + k) = x[j];
    }
  }
}

/// @brief Fill the polynomial augmentation matrix Q and sets `activeAxis`
/// accordingly to the deactivated axis in `SerialDeactivateOneAxis`
/// @tparam QViewType The polynomial matrix Q type
/// @tparam XViewType Source point view type
/// @tparam OffsViewType Access offsets for the source view type
/// @tparam WViewType Additional workspace view type
/// @tparam AxisType A `Kokkos::Array<Dim>`
/// @param Q Polynomial augmentation matrix Q
/// @param X Source points view
/// @param XOffs Access offsets for the source points views, to access the
/// points of this local cluster
/// @param W Additional workspace view
/// @param activeAxis A `Kokkos::Array` which holds the axis status
/// @return The number of active axis at the end of the function
template <KokkosViewRank<2> QViewType, KokkosViewRank<1> XViewType,
          KokkosViewRank<1> XOffsType, KokkosViewRank<1> WViewType,
          KokkosArray AxisType, int_t dim = AxisType::size()>
KOKKOS_FORCEINLINE_FUNCTION int_t SerialFillQ(QViewType &Q, const XViewType &X,
                                              const XOffsType &XOffs,
                                              WViewType &W,
                                              AxisType &activeAxis) {
  using ExecSpace = typename QViewType::execution_space;
  int_t poly_vals = dim + 1;

  Kokkos::Array<fp_t, dim + 1> scratch_data;
  for (int i = 0; i < dim + 1; ++i) {
    scratch_data[i] = fp_consts::zero();
  }
  Kokkos::View<fp_t *, ExecSpace> scratch(scratch_data.data(), dim + 1);

  do {
    SerialFillPoly(Q, X, XOffs, activeAxis);
    auto S = Kokkos::subview(scratch, Kokkos::make_pair(0, poly_vals));
    SerialSVD(Q, S, W);
    const fp_t condition =
        S(0) / Kokkos::max(S(poly_vals - 1), fp_consts::epsilon());
    if (condition > 1.0e5) {
      SerialDeactivateOneAxis(X, XOffs, activeAxis);
      --poly_vals;
    } else {
      SerialFillPoly(Q, X, XOffs, activeAxis);
      break;
    }
  } while (poly_vals > 1);
  return poly_vals;
}

/// @brief Perform: out = Ab. This function is a wrapper on `KB::SerialGemv`
/// @tparam AViewType The matrix A type
/// @tparam BViewType The vector B type
/// @tparam OutViewType The output view type
/// @param A The matrix A
/// @param B The vector B
/// @param out The output vector of out = Ab
/// @return The number of active axis at the end of the function
template <KokkosViewRank<2> AViewType, KokkosViewRank<1> BViewType,
          KokkosViewRank<1> OutViewType>
KOKKOS_FORCEINLINE_FUNCTION void
SerialMulMatVec(const AViewType &A, const BViewType &B, OutViewType &out) {
  namespace KB = KokkosBlas;
  using Gemv =
      KB::SerialGemv<KB::Trans::NoTranspose, KB::Algo::Gemv::Unblocked>;
  Gemv::invoke(fp_consts::one(), A, B, fp_consts::zero(), out);
}

/// @brief Perform an inplace addition of 2 vectors such as U = U + V
/// @tparam UViewType The vector U view type
/// @tparam VViewType The vector V view type
/// @param[in, out] U The vector U view
/// @param[in] V The vector V view
template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecAdd(UViewType &U,
                                                 const VViewType &V) {
  const int_t N = U.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    U(i) += V(i);
  }
}

/// @brief Perform an inplace difference of 2 vectors such as U = U - V
/// @tparam UViewType The vector U view type
/// @tparam VViewType The vector V view type
/// @param[in, out] U The vector U view
/// @param[in] V The vector V view
template <KokkosViewRank<1> UViewType, KokkosViewRank<1> VViewType>
KOKKOS_FORCEINLINE_FUNCTION void SerialVecVecSub(UViewType &U,
                                                 const VViewType &V) {
  const int_t N = U.extent_int(0);
  for (int_t i = 0; i < N; ++i) {
    U(i) -= V(i);
  }
}

} // namespace RbfPum
} // namespace PACMAN
