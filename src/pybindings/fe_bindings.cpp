//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#include "pybindings/fe_bindings.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "common/transfer.hxx"
#include "common/types.hpp"
#include "interpolate.hpp"
#include "pybindings/bindings.hpp"

/**
 * @file fe_bindings.cpp
 * @brief Python bindings for PACMAN finite-elements interpolation APIs.
 */

namespace PACMAN {
namespace PyBindingsFiniteElements {

/// @brief Internal typed entry point that builds a `Transfer` object,
/// dispatches interpolation, and copies results back to NumPy arrays.
/// @tparam ExecSpace Kokkos execution space selected at runtime.
/// @tparam Dim Spatial dimension of the interpolation problem.
/// @param[in] method PACMAN transfer method to execute.
/// @param[in,out] rSourcePoints Source points array `(n, Dim)`.
/// @param[in,out] rSourceValues Source scalar values array `(n)`.
/// @param[in,out] rConnVal Optional flattened connectivity values.
/// @param[in,out] rConnOff Optional connectivity offsets (CSR layout).
/// @param[in,out] rCellTypes Optional PACMAN cell types associated with
/// connectivity entries.
/// @param[in,out] rTargetPoints Target points array `(m, Dim)`.
/// @return Tuple `(target_values, transfer_status)` where both arrays have
/// length `m`.
template <typename ExecSpace, int_t Dim>
std::tuple<np_array<fp_t>, np_array<int_t>>
RunInterpolate(const method_t method, np_array<coordinates_t> &rSourcePoints,
               np_array<fp_t> &rSourceValues,
               optional_np_array<int_t> &rConnVal,
               optional_np_array<offset_t> &rConnOff,
               optional_np_array<cell_t> &rCellTypes,
               np_array<coordinates_t> &rTargetPoints) {
  auto *p_source_points =
      static_cast<coordinates_t *>(rSourcePoints.mutable_data());
  auto *p_source_values = static_cast<fp_t *>(rSourceValues.mutable_data());
  auto *p_target_points =
      static_cast<coordinates_t *>(rTargetPoints.mutable_data());
  auto *p_conn_val = static_cast<int_t *>(
      rConnVal.has_value() ? rConnVal.value().mutable_data() : nullptr);
  auto *p_conn_offsets = static_cast<offset_t *>(
      rConnOff.has_value() ? rConnOff.value().mutable_data() : nullptr);
  auto *p_cell_types = static_cast<cell_t *>(
      rCellTypes.has_value() ? rCellTypes.value().mutable_data() : nullptr);

  Transfer<ExecSpace, Dim> transfer(method);
  SetupTransferClass(transfer, rSourcePoints.shape(0),
                     rConnVal.has_value() ? rConnVal.value().shape(0) : 0,
                     rConnOff.has_value() ? rConnOff.value().shape(0) : 0,
                     rTargetPoints.shape(0), p_source_points, p_source_values,
                     p_conn_val, p_conn_offsets, p_cell_types, p_target_points);
  Interpolate(transfer);
  auto target_values_np_array = np_array<fp_t>(rTargetPoints.shape(0));
  auto transfer_status_np_array = np_array<int_t>(rTargetPoints.shape(0));
  auto *p_target_values = target_values_np_array.mutable_data();
  auto *p_transfer_status = reinterpret_cast<PACMAN::TransferStatus *>(
      transfer_status_np_array.mutable_data());
  auto unmanaged_host_tv =
      Kokkos::View<fp_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          p_target_values, rTargetPoints.shape(0));
  auto unmanaged_host_ts =
      Kokkos::View<PACMAN::TransferStatus *,
                   Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          p_transfer_status, rTargetPoints.shape(0));
  Kokkos::deep_copy(unmanaged_host_tv, transfer.targetValues);
  Kokkos::deep_copy(unmanaged_host_ts, transfer.targetStatus);
  return std::make_tuple(target_values_np_array, transfer_status_np_array);
}

/// @brief This function allows us to pass runtime parameters as template
/// arguments, using a type union `std::variant` (`AvailableExecSpaces`)
/// @tparam Dim The space dimension of the problem, here the dimension is
/// checked to be between 1 and 3 (included)
/// @note The pattern using a `std::variant` and `std::visit` has multiple
/// advantages: it allows us to dispatch safely the runtime arguments to
/// templated function, with no big switch or polymorphism (which can be
/// confusing), also, we are sure that all of the cases are handled properly, as
/// we won't be able to compule the code until all the union values have their
/// value handled.
///       In short: it is safer, implies no heap allocation, no virtual dispatch
///       and allows the compiler to check and optimise everything properly
template <int_t Dim>
requires IsValidDim<Dim> // 1 <= Dim <= 3
    std::tuple<np_array<fp_t>, np_array<int_t>>
    Dispatch(const unsigned char execSpace, const method_t method,
             np_array<coordinates_t> &rSourcePoints,
             np_array<fp_t> &rSourceValues, optional_np_array<int_t> &rConnVal,
             optional_np_array<offset_t> &rConnOff,
             optional_np_array<cell_t> &rCellTypes,
             np_array<coordinates_t> &rTargetPoints) {
  return std::visit(
      [&](auto execSpaceObj) {
        return RunInterpolate<decltype(execSpaceObj), Dim>(
            method, rSourcePoints, rSourceValues, rConnVal, rConnOff,
            rCellTypes, rTargetPoints);
      },
      MakeExecSpace(execSpace));
}

/// @brief Python-facing finite-elements interpolation binding.
///
/// Validates NumPy input arrays, dispatches the call according to
/// `spaceDimension` and `execSpace`, and returns interpolated values plus
/// transfer status for each target point.
///
/// @param[in] spaceDimension Geometric dimension of points (`1`, `2`, or `3`).
/// @param[in] execSpace Runtime execution-space selector.
/// @param[in] method Transfer method selector from PACMAN FE methods.
/// @param[in] sourcePoints Source points array shaped `(n, spaceDimension)`.
/// @param[in] sourceValues Source scalar values array shaped `(n)`.
/// @param[in] connVal Optional flattened connectivity values.
/// @param[in] connOff Optional connectivity offsets (CSR layout).
/// @param[in] cellTypes Optional PACMAN cell type identifiers.
/// @param[in] targetPoints Target points array shaped `(m, spaceDimension)`.
/// @return Tuple `(target_values, transfer_status)` where both arrays have
/// length `m`.
/// @throws std::invalid_argument when array dimensions/shapes are inconsistent.
/// @throws std::runtime_error when `spaceDimension` is not in `{1,2,3}`.
std::tuple<np_array<fp_t>, np_array<int_t>>
Interpolate(const int_t spaceDimension, const unsigned char execSpace,
            const method_t method, np_array<coordinates_t> sourcePoints,
            np_array<fp_t> sourceValues, optional_np_array<int_t> connVal,
            optional_np_array<offset_t> connOff,
            optional_np_array<cell_t> cellTypes,
            np_array<coordinates_t> targetPoints) {
  const std::string _region_name = "PyBindingsFiniteElements::Interpolate";
  const Kokkos::Profiling::ScopedRegion region(_region_name);

  if (sourcePoints.ndim() != 2) {
    throw std::invalid_argument("Source points must be a 2-d numpy array!");
  }
  if (targetPoints.ndim() != 2) {
    throw std::invalid_argument("Target points must be a 2-d numpy array!");
  }
  if (sourcePoints.shape(1) != targetPoints.shape(1)) {
    throw std::invalid_argument(
        "Dimensions mismatch between source points and target "
        "points!\nProvided source points dimension: " +
        std::to_string(sourcePoints.shape(1)) +
        "\nProvided target points dimension: " +
        std::to_string(targetPoints.shape(1)) + "\n");
  }
  if (sourcePoints.shape(0) != sourceValues.shape(0)) {
    throw std::invalid_argument(
        "Each source point must have its corresponding value in "
        "the "
        "source values!\nNumber of source points: " +
        std::to_string(sourcePoints.shape(0)) + "\nNumber of source values: " +
        std::to_string(sourceValues.shape(0)) + "\n");
  }

  switch (spaceDimension) {
  case 1:
    return Dispatch<1>(execSpace, method, sourcePoints, sourceValues, connVal,
                       connOff, cellTypes, targetPoints);
  case 2:
    return Dispatch<2>(execSpace, method, sourcePoints, sourceValues, connVal,
                       connOff, cellTypes, targetPoints);
  case 3:
    return Dispatch<3>(execSpace, method, sourcePoints, sourceValues, connVal,
                       connOff, cellTypes, targetPoints);
  default:
    throw new std::runtime_error(
        "The dimension of the points can only be: 1, 2 or 3.\n");
  }
}

/// @brief Convert VTK/vtk cell type identifiers to PACMAN cell type codes.
/// @param[in] cell_type 1D NumPy array of VTK cell type identifiers.
/// @return 1D NumPy array of PACMAN cell type identifiers with same length.
/// @throws std::runtime_error if at least one VTK cell type is unsupported.
np_array<int_t> vtk_to_pacman_cell_type(const np_array<int_t> cell_type) {
  static const std::unordered_map<int_t, PACMAN::CellType> vtk_to_pacman = {
      // 0D
      {1, PACMAN::CellType::VTK_VERTEX},              // VTK_VERTEX
                                                      // 1D
      {3, PACMAN::CellType::VTK_LINE},                // VTK_LINE
      {21, PACMAN::CellType::VTK_QUADRATIC_EDGE},     // VTK_QUADRATIC_EDGE
                                                      // 2D
      {5, PACMAN::CellType::VTK_TRIANGLE},            // VTK_TRIANGLE
      {9, PACMAN::CellType::VTK_QUAD},                // VTK_QUAD
      {22, PACMAN::CellType::VTK_QUADRATIC_TRIANGLE}, // VTK_QUADRATIC_TRIANGLE
      {23, PACMAN::CellType::VTK_QUADRATIC_QUAD},     // VTK_QUADRATIC_QUAD
                                                      // 3D
      {10, PACMAN::CellType::VTK_TETRA},              // VTK_TETRA
      {12, PACMAN::CellType::VTK_HEXAHEDRON},         // VTK_HEXAHEDRON
      {13, PACMAN::CellType::VTK_WEDGE},              // VTK_WEDGE
      {14, PACMAN::CellType::VTK_PYRAMID},            // VTK_PYRAMID
      {24, PACMAN::CellType::VTK_QUADRATIC_TETRA},    // VTK_QUADRATIC_TETRA
      {25,
       PACMAN::CellType::VTK_QUADRATIC_HEXAHEDRON},  // VTK_QUADRATIC_HEXAHEDRON
      {26, PACMAN::CellType::VTK_QUADRATIC_WEDGE},   // VTK_QUADRATIC_WEDGE
      {27, PACMAN::CellType::VTK_QUADRATIC_PYRAMID}, // VTK_QUADRATIC_PYRAMID
  };

  auto pacman_cell_type_np_array = np_array<int_t>(cell_type.shape(0));
  auto *p_pct = pacman_cell_type_np_array.mutable_data();
  auto unmanaged_pct =
      Kokkos::View<int_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          p_pct, pacman_cell_type_np_array.shape(0));

  auto ct_buf = cell_type.unchecked<1>();

  Kokkos::parallel_for(
      "vtk celltype to pacman celltype",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
          Kokkos::DefaultHostExecutionSpace{}, 0,
          pacman_cell_type_np_array.shape(0)),
      [=](const index_t &i) {
        auto it = vtk_to_pacman.find(ct_buf(i));
        if (it == vtk_to_pacman.end()) {
          throw std::runtime_error("Unsupported VTK cell type: " +
                                   std::to_string(ct_buf(i)));
        }
        unmanaged_pct(i) =
            static_cast<int_t>(static_cast<cell_t>((it->second)));
      });

  return pacman_cell_type_np_array;
}

/// @brief Return the topological dimension associated with one VTK cell type.
/// @param[in] cell_type Single VTK/vtk cell type identifier.
/// @return Cell dimension (`0`, `1`, `2`, or `3`).
/// @throws std::runtime_error if the given cell type is unsupported.
int_t vtk_cell_dim(const int_t &cell_type) {
  static const std::unordered_map<int_t, int_t> vtk_dim = {
      //== 0D
      {1, 0}, // VTK_VERTEX
      //== 1D
      {3, 1},  // VTK_LINE
      {21, 1}, // VTK_QUADRATIC_EDGE
      //== 2D
      {5, 2},  // VTK_TRIANGLE
      {9, 2},  // VTK_QUAD
      {22, 2}, // VTK_QUADRATIC_TRIANGLE
      {23, 2}, // VTK_QUADRATIC_QUAD
      //== 3D
      {10, 3}, // VTK_TETRA
      {12, 3}, // VTK_HEXAHEDRON
      {13, 3}, // VTK_WEDGE
      {14, 3}, // VTK_PYRAMID
      {24, 3}, // VTK_QUADRATIC_TETRA
      {25, 3}, // VTK_QUADRATIC_HEXAHEDRON
      {26, 3}, // VTK_QUADRATIC_WEDGE
      {27, 3}, // VTK_QUADRATIC_PYRAMID
  };

  auto it = vtk_dim.find(cell_type);
  if (it == vtk_dim.end()) {
    throw std::runtime_error("Unsupported Vtk cell type: " +
                             std::to_string(cell_type));
  }
  return it->second;
}

} // namespace PyBindingsFiniteElements

} // namespace PACMAN
