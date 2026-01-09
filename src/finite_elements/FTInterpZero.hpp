//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

// TODO:

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FTUtils.hpp"

namespace FiniteElement {

template <typename ExecSpace, int spaceDimension>
/**
 * @brief Compute the target point finite element interpolation if intersects or
 * fill with zero value if outside all elements.
 * @param[in] transfer the transfer class holding informations.
 * @return sparse coo matrix containing interpolation coefficient aka field
 * transfer operator
 */
SparsedMatCOO
FTInterZero(TransferClassKokkos<ExecSpace, spaceDimension> *transfer) {
  Kokkos::Profiling::pushRegion("FTInterZero");

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

  // Can we avoid this reduction and branching
  int nnz = 0;
  Kokkos::Profiling::pushRegion("Get nnz");
  Kokkos::parallel_reduce(
      "Compute COO operator functor",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbtargetPoints),
      KOKKOS_LAMBDA(int i, int &nbEntries) {
        if (targetStatusPtr(i) == TransferStatus::INTER) {
          nbEntries += countEntries(i);
        } else { // set zeros
          nbEntries++;
          col2d_d(i, 0) = 0; // this is arbitrary as the value is zero
          countEntries(i) = 1;
          targetStatusPtr(i) = TransferStatus::ZERO_FILL;
          targetElemPtr(i) = 0; // this is arbitrary as the value is zero
          val2d_d(i, 0) = 0.0;
        }
      },
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
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::deep_copy(matCOO.rows, rows_d);
  Kokkos::deep_copy(matCOO.cols, cols_d);
  Kokkos::deep_copy(matCOO.vals, vals_d);

  Kokkos::Profiling::popRegion();
  return matCOO;
}
} // namespace FiniteElement
