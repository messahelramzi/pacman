//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <Kokkos_Abort.hpp>
#include <stdexcept>

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FETools.hpp"
#include "finite_elements/utils/NewtonKokkos.hpp"

namespace PACMAN {

namespace FiniteElements {

template <typename ExecSpace, int_t Dim>
/**
 * @brief Intersect target points with source-element bounding boxes, then
 * evaluate FE interpolation for intersecting candidates.
 * @tparam ExecSpace Kokkos execution space used for search/interpolation.
 * @tparam Dim Spatial dimension of the interpolation problem.
 * @param[in,out] transfer Transfer descriptor containing source mesh,
 * connectivity, source values, and target buffers.
 *
 * Reads: `sourcePoints`, `sourceValues`, `connValues`, `connOffsets`,
 * `cellTypes`.
 * Writes: `targetValues`, `targetStatus` (for points successfully
 * interpolated as `TransferStatus::INTER`).
 */
void ComputeBoxTargetPointIntersection(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion(
      "FiniteElement::ComputeBoxTargetPointIntersection");
  if (transfer.nbSpaceDimElements == 0) {
    return;
  }

  using MemorySpace = typename ExecSpace::memory_space;
  using Box = ArborX::Box<Dim, coordinates_t>;

  ExecSpace execSpace{};

  Kokkos::Profiling::pushRegion(
      "copy pointers and allocate auxiliary data structures");

  auto sourcePointsPtr = transfer.sourcePoints;
  auto sourceValuesPtr = transfer.sourceValues;

  auto targetPointsPtr = transfer.targetPoints;
  auto targetValuesPtr = transfer.targetValues;
  auto targetStatusPtr = transfer.targetStatus;
  auto nbtargetPoints = targetPointsPtr.extent_int(0);

  auto connValPtr = transfer.connValues;
  auto connOffPtr = transfer.connOffsets;
  auto CellTypesPtr = transfer.cellTypes;

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion(
      "Allocate auxiliary data structures for box-point intersection");
  Kokkos::View<Box *, MemorySpace> bboxes(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Boundary boxes of elements"),
      transfer.nbSpaceDimElements);
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Parents of Bboxes"),
      transfer.nbSpaceDimElements);
  Kokkos::Profiling::popRegion();

  // Kokkos::parallel_scan(
  //     "Compute element bounding boxes functor",
  //     Kokkos::RangePolicy(execSpace, 0, transfer.nbElems),
  //     KOKKOS_LAMBDA(const int_t &i, int_t &partial_sum, bool is_final) {
  //       const CellType cellType = static_cast<CellType>(CellTypesPtr(i));
  //       const int_t elemToTreat =
  //           static_cast<int_t>(getDimension(cellType) == Dim);
  //       const int_t firstIndex = partial_sum;
  //       partial_sum += elemToTreat;
  //       if (is_final) {
  //         for (int_t j = firstIndex; j < firstIndex + elemToTreat; j++) {
  //           bboxes(j) = Box();
  //           auto localConn = Kokkos::subview(
  //               connValPtr,
  //               Kokkos::make_pair(connOffPtr(i), connOffPtr(i + 1)));
  //           const int_t nbConnNodes = connOffPtr(i + 1) - connOffPtr(i);
  //           for (int_t k = 0; k < nbConnNodes; k++) {
  //             ArborX::Details::expand(bboxes(j),
  //             sourcePointsPtr(localConn(k)));
  //           }
  //           parents(j) = i;
  //         }
  //       }
  //     });

  Kokkos::parallel_for(
      "Compute element bounding boxes functor",
      Kokkos::RangePolicy(execSpace, 0, transfer.nbElems),
      KOKKOS_LAMBDA(const int_t &i) {
        ArborX::Point<Dim, coordinates_t> pmin;
        ArborX::Point<Dim, coordinates_t> pmax;
        for (int d = 0; d < Dim; ++d) {
          pmin[d] = 9999999.0;
          pmax[d] = -9999999.0;
        }
        for (int_t k = connOffPtr(i); k < connOffPtr(i + 1); k++) {
          for (int d = 0; d < Dim; ++d) {
            pmin[d] = Kokkos::min(pmin[d], sourcePointsPtr(connValPtr(k))[d]);
            pmax[d] = Kokkos::max(pmax[d], sourcePointsPtr(connValPtr(k))[d]);
          }
        }
        bboxes(i) = Box(pmin, pmax);
        parents(i) = i;
      });

  ArborX::BoundingVolumeHierarchy bvhBoxes(
      execSpace, ArborX::Experimental::attach_indices(bboxes));

  PointIntersect<MemorySpace, Dim> pi{targetPointsPtr};
  bvhBoxes.query(
      execSpace, pi,
      PointFiniteElementInterpolation<ExecSpace, Dim>{transfer, parents});

  Kokkos::Profiling::popRegion(); // ComputeBoxTargetPointIntersection
}

} // namespace FiniteElements
} // namespace PACMAN
