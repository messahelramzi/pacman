#pragma once

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>

#include "common/types.hpp"
#include "finite_elements/utils/FEInterpolatorTools.hxx"

namespace PACMAN {
static constexpr int_t MaxNodesPerElt = 27;

template <typename ExecSpace, int_t Dim> struct Transfer {
  using MemorySpace = typename ExecSpace::memory_space;

  TransferMethods method;

  Kokkos::View<::ArborX::Point<Dim, coordinates_t> *, MemorySpace> sourcePoints;
  Kokkos::View<fp_t *, MemorySpace> sourceValues;

  Kokkos::View<coordinates_t **, MemorySpace> targetPoints;
  Kokkos::View<fp_t *, MemorySpace> targetValues;
  Kokkos::View<TransferStatus *, MemorySpace> targetStatus;

  int_t nbElems;
  int_t nbSpaceDimElements = 0;
  Kokkos::View<int_t *, MemorySpace> connValues;
  Kokkos::View<offset_t *, MemorySpace> connOffsets;
  Kokkos::View<cell_t *, MemorySpace> cellTypes;

  int_t nbLinearSkinFaces;
  Kokkos::View<int_t **, MemorySpace> skinFaces;
  Kokkos::View<int_t *, MemorySpace> skinParents;

  Transfer(const method_t method) {
    this->method = static_cast<TransferMethods>(method);
  }
  Transfer(const TransferMethods method) { this->method = method; }
};

template <typename ExecSpace, int_t Dim>
void SetupTransferClass(Transfer<ExecSpace, Dim> &transfer,
                        const index_t sourcePointsSize,
                        const index_t connValSize, const index_t connOffSize,
                        const index_t targetPointsSize,
                        coordinates_t *pSourcePoints, fp_t *pSourceValues,
                        int_t *pConnVal, offset_t *pConnOff, cell_t *pCellTypes,
                        coordinates_t *pTargetPoints) {
  const std::string _region_name = "SetupTransferClass";
  using MemorySpace = typename ExecSpace::memory_space;

  ExecSpace execspace{};
  Kokkos::DefaultHostExecutionSpace hostspace{};

  auto unmanaged_host_tp =
      Kokkos::View<coordinates_t **,
                   Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          pTargetPoints, targetPointsSize, Dim);
  auto unmanaged_device_tp =
      Kokkos::create_mirror_view_and_copy(execspace, unmanaged_host_tp);
  transfer.targetPoints = Kokkos::View<coordinates_t **, MemorySpace>(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "transfer.targetPoints"),
      targetPointsSize, Dim);

  transfer.targetValues = Kokkos::View<fp_t *, MemorySpace>(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "transfer.targetValues"),
      targetPointsSize);

  transfer.targetStatus = Kokkos::View<TransferStatus *, MemorySpace>(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "transfer.targetStatus"),
      targetPointsSize);

  auto target_points = transfer.targetPoints;
  auto target_status = transfer.targetStatus;
  auto target_values = transfer.targetValues;

  Kokkos::parallel_for(
      _region_name + "::copy target points",
      Kokkos::RangePolicy<ExecSpace>(execspace, 0, targetPointsSize),
      KOKKOS_LAMBDA(const index_t &i) {
        target_status(i) = TransferStatus::OUTSIDE;
        for (int_t j = 0; j < Dim; j++) {
          target_points(i, j) = unmanaged_device_tp(i, j);
        }
        target_values(i) = fp_consts::zero();
      });

  ExecSpace execspace2{};
  auto unmanaged_host_sp =
      Kokkos::View<coordinates_t **,
                   Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          pSourcePoints, sourcePointsSize, Dim);
  auto unmanaged_device_sp =
      Kokkos::create_mirror_view_and_copy(execspace2, unmanaged_host_sp);
  transfer.sourcePoints =
      Kokkos::View<::ArborX::Point<Dim, coordinates_t> *, ExecSpace>(
          Kokkos::view_alloc(execspace2, Kokkos::WithoutInitializing,
                             "transfer.sourcePoints"),
          sourcePointsSize);

  auto unmanaged_host_sv =
      Kokkos::View<fp_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(pSourceValues,
                                                            sourcePointsSize);
  auto unmanaged_device_sv =
      Kokkos::create_mirror_view_and_copy(execspace2, unmanaged_host_sv);
  transfer.sourceValues = Kokkos::View<fp_t *, ExecSpace>(
      Kokkos::view_alloc(execspace2, Kokkos::WithoutInitializing,
                         "transfer.sourceValues"),
      sourcePointsSize);

  auto source_points = transfer.sourcePoints;
  auto source_values = transfer.sourceValues;
  Kokkos::parallel_for(
      _region_name + "::copy source nodes",
      Kokkos::RangePolicy(execspace2, 0, sourcePointsSize),
      KOKKOS_LAMBDA(const index_t &i) {
        for (int_t j = 0; j < Dim; j++) {
          source_points(i)[j] = unmanaged_device_sp(i, j);
        }
        source_values(i) = unmanaged_device_sv(i);
      });

  transfer.nbLinearSkinFaces = -99;
  transfer.nbSpaceDimElements = -99;
  // No need of conn for these methods
  if (transfer.method != TransferMethods::NEAREST_NEAREST &&
      transfer.method != TransferMethods::RBF_PUM) {

    auto unmanagedHost_cv =
        Kokkos::View<int_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(pConnVal,
                                                              connValSize);
    transfer.connValues =
        Kokkos::create_mirror_view_and_copy(execspace2, unmanagedHost_cv);

    auto unmanagedHost_co =
        Kokkos::View<int_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(pConnOff,
                                                              connOffSize);
    transfer.connOffsets =
        Kokkos::create_mirror_view_and_copy(execspace2, unmanagedHost_co);

    transfer.nbElems = connOffSize - 1;
    auto unmanagedHost_ct =
        Kokkos::View<int_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(pCellTypes,
                                                              transfer.nbElems);
    transfer.cellTypes =
        Kokkos::create_mirror_view_and_copy(execspace2, unmanagedHost_ct);

    auto connOffsetsPtr = transfer.connOffsets;
    auto cellTypesPtr = transfer.cellTypes;

    int_t result1;
    int_t result2;

    Kokkos::parallel_reduce(
        "Transfer connectivity data",
        Kokkos::RangePolicy<ExecSpace>(execspace2, 0, transfer.nbElems),
        KOKKOS_LAMBDA(const int_t &i, int_t &lsum1, int_t &lsum2) {
          const CellType cellType = static_cast<CellType>(cellTypesPtr(i));
          lsum1 += FiniteElements::getNbFace<Dim>(cellType);
          lsum2 += (FiniteElements::getDimension(cellType) == Dim);
        },
        Kokkos::Sum<int_t>(result1), Kokkos::Sum<int_t>(result2));
    transfer.nbLinearSkinFaces = result1;
    transfer.nbSpaceDimElements = result2;
  }
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // SetupTransferClass
}
} // namespace PACMAN
