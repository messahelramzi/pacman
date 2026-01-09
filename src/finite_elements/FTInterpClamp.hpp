//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/ComputeLinearSkin.hxx"
#include "finite_elements/utils/FTUtils.hpp"
#include "finite_elements/utils/SkinProjection.hpp"

namespace FiniteElement {

template <typename ExecSpace, int spaceDimension>
/**
 * @brief Compute the target point finite element interpolation if intersects or
 * assign nearest element interpolation values after projection of the target
 * point if outside all elements.
 * @param[in] transfer the transfer class holding informations.
 * @return sparse coo matrix containing interpolation coefficient aka field
 * transfer operator
 */
SparsedMatCOO
FTInterClamp(TransferClassKokkos<ExecSpace, spaceDimension> *transfer) {
  Kokkos::Profiling::pushRegion("FTInterClamp");

  using MemorySpace = typename ExecSpace::memory_space;
  using Box = ArborX::Box<spaceDimension, FPType>;

  ExecSpace execSpace{};

  auto targetPointsPtr = transfer->targetPoints;
  auto targetElemPtr = transfer->targetElement;
  auto targetStatusPtr = transfer->targetStatus;
  auto elementsTypePtr = transfer->CellTypes;
  auto nodesPtr = transfer->nodes;
  auto maxConnSizePerElt = transfer->maxConnSizePerElt;

  auto nbtargetPoints = targetPointsPtr.extent(0);

  Kokkos::Profiling::pushRegion("Compute target point FE intersection");
  auto [col2d_d, val2d_d, countEntries] =
      ComputeBoxTargetPointIntersection<ExecSpace, spaceDimension>(transfer);
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion(
      "Compute target point projection on triangulated skin mesh");
  ComputeLinearSkin<ExecSpace, spaceDimension>(transfer);
  if constexpr (spaceDimension == 1)
    ComputeProjectionOn1DSkin<ExecSpace>(transfer, col2d_d, val2d_d,
                                         countEntries, false);
  else if constexpr (spaceDimension == 2)
    ComputeProjectionOn2DSkin<ExecSpace>(transfer, col2d_d, val2d_d,
                                         countEntries, false);
  else if constexpr (spaceDimension == 3)
    ComputeProjectionOn3DSkin<ExecSpace>(transfer, col2d_d, val2d_d,
                                         countEntries, false);
  else
    throw std::runtime_error("Unexpected dimension for projection");
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  // Can we avoid this reduction and branching
  int nnz = 0;
  Kokkos::Profiling::pushRegion("Get nnz");
  Kokkos::parallel_reduce(
      "Compute COO operator functor",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(int i, int &nbEntries) { nbEntries += countEntries(i); },
      nnz);
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  // Scan (assemble) COO operator !
  Kokkos::Profiling::pushRegion("Scan COO operator");
  auto matCOO = SparsedMatCOO(nnz);
  auto rows_d = Kokkos::View<INTType *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "row_d"), nnz);
  auto cols_d = Kokkos::View<INTType *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "col_d"), nnz);
  auto vals_d = Kokkos::View<FPType *, MemorySpace>(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "data_d"),
      nnz);
  Kokkos::parallel_scan(
      "Scan COO operator functor",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(int64_t i, int64_t &partial_sum, bool is_final) {
        auto firstIndex = partial_sum;
        partial_sum += countEntries(i);
        if (is_final) {
          for (int64_t j = 0; j < countEntries(i); j++) {
            rows_d(firstIndex + j) = i;
            cols_d(firstIndex + j) = col2d_d(i, j);
            vals_d(firstIndex + j) = val2d_d(i, j);
          }
        }
      });
  Kokkos::Profiling::popRegion();
  Kokkos::fence();

  Kokkos::deep_copy(matCOO.rows, rows_d);
  Kokkos::deep_copy(matCOO.cols, cols_d);
  Kokkos::deep_copy(matCOO.vals, vals_d);

  Kokkos::Profiling::popRegion();
  return matCOO;
}
} // namespace FiniteElement
