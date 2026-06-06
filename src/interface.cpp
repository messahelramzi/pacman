//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

// Define before including interpolate.hpp so the extern-template suppression
// block is skipped and this TU can provide explicit instantiations.
#define PACMAN_FE_ETI_COMPILATION

#include "interface.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "interpolate.hpp"

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

template <typename ExecSpace, PACMAN::int_t Dim>
PACMAN::FeInterpolateResult
RunInterpolate(PACMAN::method_t method, PACMAN::coordinates_t *pSourcePoints,
               PACMAN::int_t nSourcePoints, PACMAN::fp_t *pSourceValues,
               PACMAN::int_t *pConnVal, PACMAN::int_t connValSize,
               PACMAN::offset_t *pConnOff, PACMAN::int_t connOffSize,
               PACMAN::cell_t *pCellTypes, PACMAN::coordinates_t *pTargetPoints,
               PACMAN::int_t nTargetPoints, bool fortranIndexing) {
  PACMAN::Transfer<ExecSpace, Dim> transfer(method);
  PACMAN::SetupTransferClass(transfer, nSourcePoints, connValSize, connOffSize,
                             nTargetPoints, pSourcePoints, pSourceValues,
                             pConnVal, pConnOff, pCellTypes, pTargetPoints, fortranIndexing);
  PACMAN::Interpolate(transfer);

  PACMAN::FeInterpolateResult result;
  result.targetValues.resize(nTargetPoints);
  result.targetStatus.resize(nTargetPoints);

  auto unmanaged_host_tv =
      Kokkos::View<PACMAN::fp_t *,
                   Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          result.targetValues.data(), nTargetPoints);
  auto unmanaged_host_ts = Kokkos::View<
      PACMAN::TransferStatus *, Kokkos::DefaultHostExecutionSpace::memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      reinterpret_cast<PACMAN::TransferStatus *>(result.targetStatus.data()),
      nTargetPoints);

  Kokkos::deep_copy(unmanaged_host_tv, transfer.targetValues);
  Kokkos::deep_copy(unmanaged_host_ts, transfer.targetStatus);
  return result;
}

template <PACMAN::int_t Dim>
requires PACMAN::IsValidDim<Dim> PACMAN::FeInterpolateResult
Dispatch(unsigned char execSpace, PACMAN::method_t method,
         PACMAN::coordinates_t *pSourcePoints, PACMAN::int_t nSourcePoints,
         PACMAN::fp_t *pSourceValues, PACMAN::int_t *pConnVal,
         PACMAN::int_t connValSize, PACMAN::offset_t *pConnOff,
         PACMAN::int_t connOffSize, PACMAN::cell_t *pCellTypes,
         PACMAN::coordinates_t *pTargetPoints, PACMAN::int_t nTargetPoints, bool fortranIndexing) {
  return std::visit(
      [&](auto execSpaceObj) {
        return RunInterpolate<decltype(execSpaceObj), Dim>(
            method, pSourcePoints, nSourcePoints, pSourceValues, pConnVal,
            connValSize, pConnOff, connOffSize, pCellTypes, pTargetPoints,
            nTargetPoints, fortranIndexing);
      },
      PACMAN::MakeExecSpace(execSpace));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

namespace PACMAN {

FeInterpolateResult
fe_interpolate(int_t spaceDimension, unsigned char execSpace, method_t method,
               coordinates_t *sourcePoints, int_t nSourcePoints,
               fp_t *sourceValues, int_t *connVal, int_t connValSize,
               offset_t *connOff, int_t connOffSize, cell_t *cellTypes,
               coordinates_t *targetPoints, int_t nTargetPoints, bool fortranIndexing) {
  const std::string _region_name = "PACMAN::fe_interpolate";
  const Kokkos::Profiling::ScopedRegion region(_region_name);

  if (sourcePoints == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourcePoints is null but nSourcePoints > 0");
  if (targetPoints == nullptr && nTargetPoints > 0)
    throw std::invalid_argument("targetPoints is null but nTargetPoints > 0");
  if (sourceValues == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourceValues is null but nSourcePoints > 0");

  switch (spaceDimension) {
  case 1:
    return Dispatch<1>(execSpace, method, sourcePoints, nSourcePoints,
                       sourceValues, connVal, connValSize, connOff, connOffSize,
                       cellTypes, targetPoints, nTargetPoints, fortranIndexing);
  case 2:
    return Dispatch<2>(execSpace, method, sourcePoints, nSourcePoints,
                       sourceValues, connVal, connValSize, connOff, connOffSize,
                       cellTypes, targetPoints, nTargetPoints, fortranIndexing);
  case 3:
    return Dispatch<3>(execSpace, method, sourcePoints, nSourcePoints,
                       sourceValues, connVal, connValSize, connOff, connOffSize,
                       cellTypes, targetPoints, nTargetPoints, fortranIndexing);
  default:
    throw std::runtime_error(
        "The dimension of the points can only be: 1, 2 or 3.\n");
  }
}

void vtk_to_pacman_cell_type(const int_t *vtkTypes, cell_t *pacmanTypes,
                             int_t n) {
  static const std::unordered_map<int_t, CellType> vtk_to_pacman = {
      // 0D
      {1, CellType::VTK_VERTEX},
      // 1D
      {3, CellType::VTK_LINE},
      {21, CellType::VTK_QUADRATIC_EDGE},
      // 2D
      {5, CellType::VTK_TRIANGLE},
      {9, CellType::VTK_QUAD},
      {22, CellType::VTK_QUADRATIC_TRIANGLE},
      {23, CellType::VTK_QUADRATIC_QUAD},
      // 3D
      {10, CellType::VTK_TETRA},
      {12, CellType::VTK_HEXAHEDRON},
      {13, CellType::VTK_WEDGE},
      {14, CellType::VTK_PYRAMID},
      {24, CellType::VTK_QUADRATIC_TETRA},
      {25, CellType::VTK_QUADRATIC_HEXAHEDRON},
      {26, CellType::VTK_QUADRATIC_WEDGE},
      {27, CellType::VTK_QUADRATIC_PYRAMID},
  };
  for (int_t i = 0; i < n; ++i) {
    auto it = vtk_to_pacman.find(vtkTypes[i]);
    if (it == vtk_to_pacman.end())
      throw std::runtime_error("Unsupported VTK cell type: " +
                               std::to_string(vtkTypes[i]));
    pacmanTypes[i] = static_cast<cell_t>(it->second);
  }
}

int_t vtk_cell_dim(int_t vtkCellType) {
  static const std::unordered_map<int_t, int_t> vtk_dim = {
      // 0D
      {1, 0},
      // 1D
      {3, 1},
      {21, 1},
      // 2D
      {5, 2},
      {9, 2},
      {22, 2},
      {23, 2},
      // 3D
      {10, 3},
      {12, 3},
      {13, 3},
      {14, 3},
      {24, 3},
      {25, 3},
      {26, 3},
      {27, 3},
  };
  auto it = vtk_dim.find(vtkCellType);
  if (it == vtk_dim.end())
    throw std::runtime_error("Unsupported VTK cell type: " +
                             std::to_string(vtkCellType));
  return it->second;
}

} // namespace PACMAN
