#include "pybindings/fe_bindings.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "common/transfer.hxx"
#include "common/types.hpp"
#include "interpolate.hpp"
#include "pybindings/bindings.hpp"

namespace PACMAN {
namespace PyBindingsFiniteElements {
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

template <int_t Dim>
  requires IsValidDim<Dim> // 1 <= Dim <= 3
std::tuple<np_array<fp_t>, np_array<int_t>>
Dispatch(const unsigned char execSpace, const method_t method,
         np_array<coordinates_t> &rSourcePoints, np_array<fp_t> &rSourceValues,
         optional_np_array<int_t> &rConnVal,
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

int_t meshio_to_vtk_cell_type(const std::string &cell_type) {
  static const std::unordered_map<std::string, PACMAN::CellType> meshio_to_vtk =
      {
          // 0D
          {"vertex", PACMAN::CellType::VTK_VERTEX},        // VTK_VERTEX
                                                           // 1D
          {"line", PACMAN::CellType::VTK_LINE},            // VTK_LINE
          {"line3", PACMAN::CellType::VTK_QUADRATIC_EDGE}, // VTK_QUADRATIC_EDGE
                                                           // 2D
          {"triangle", PACMAN::CellType::VTK_TRIANGLE},    // VTK_TRIANGLE
          {"quad", PACMAN::CellType::VTK_QUAD},            // VTK_QUAD
          {"triangle6",
           PACMAN::CellType::VTK_QUADRATIC_TRIANGLE}, // VTK_QUADRATIC_TRIANGLE
          {"quad8", PACMAN::CellType::VTK_QUADRATIC_QUAD}, // VTK_QUADRATIC_QUAD
                                                           // 3D
          {"tetra", PACMAN::CellType::VTK_TETRA},          // VTK_TETRA
          {"hexahedron", PACMAN::CellType::VTK_HEXAHEDRON}, // VTK_HEXAHEDRON
          {"wedge", PACMAN::CellType::VTK_WEDGE},           // VTK_WEDGE
          {"pyramid", PACMAN::CellType::VTK_PYRAMID},       // VTK_PYRAMID
          {"tetra10",
           PACMAN::CellType::VTK_QUADRATIC_TETRA}, // VTK_QUADRATIC_TETRA
          {"hexahedron20",
           PACMAN::CellType::
               VTK_QUADRATIC_HEXAHEDRON}, // VTK_QUADRATIC_HEXAHEDRON
          {"wedge15",
           PACMAN::CellType::VTK_QUADRATIC_WEDGE}, // VTK_QUADRATIC_WEDGE
          {"pyramid13",
           PACMAN::CellType::VTK_QUADRATIC_PYRAMID}, // VTK_QUADRATIC_PYRAMID
      };

  auto it = meshio_to_vtk.find(cell_type);
  if (it == meshio_to_vtk.end()) {
    throw std::runtime_error("Unsupported meshio cell type: " + cell_type);
  }
  return static_cast<int_t>(static_cast<cell_t>((it->second)));
}

int_t meshio_cell_dim(const std::string &cell_type) {
  static const std::unordered_map<std::string, int_t> meshio_dim = {
      // 0D
      {"vertex", 0}, // VTK_VERTEX
      // 1D
      {"line", 1},  // VTK_LINE
      {"line3", 1}, // VTK_QUADRATIC_EDGE
      // 2D
      {"triangle", 2},  // VTK_TRIANGLE
      {"quad", 2},      // VTK_QUAD
      {"triangle6", 2}, // VTK_QUADRATIC_TRIANGLE
      {"quad8", 2},     // VTK_QUADRATIC_QUAD
      // 3D
      {"tetra", 3},        // VTK_TETRA
      {"hexahedron", 3},   // VTK_HEXAHEDRON
      {"wedge", 3},        // VTK_WEDGE
      {"pyramid", 3},      // VTK_PYRAMID
      {"tetra10", 3},      // VTK_QUADRATIC_TETRA
      {"hexahedron20", 3}, // VTK_QUADRATIC_HEXAHEDRON
      {"wedge15", 3},      // VTK_QUADRATIC_WEDGE
      {"pyramid13", 3},    // VTK_QUADRATIC_PYRAMID
  };

  auto it = meshio_dim.find(cell_type);
  if (it == meshio_dim.end()) {
    throw std::runtime_error("Unsupported meshio cell type: " + cell_type);
  }
  return it->second;
}

} // namespace PyBindingsFiniteElements

} // namespace PACMAN
