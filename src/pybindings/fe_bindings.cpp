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
py::array_t<fp_t> RunInterpolate(const method_t method,
                                 np_array<coordinates_t> &rSourcePoints,
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
  auto target_values_np_array = py::array_t<fp_t>(rTargetPoints.shape(0));
  auto *p_target_values = target_values_np_array.mutable_data();
  auto unmanaged_host_tv =
      Kokkos::View<fp_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          p_target_values, rTargetPoints.shape(0));
  Kokkos::deep_copy(unmanaged_host_tv, transfer.targetValues);
  return target_values_np_array;
}

template <int Dim>
py::array_t<fp_t> Dispatch(const unsigned char execSpace, const method_t method,
                           np_array<coordinates_t> &rSourcePoints,
                           np_array<fp_t> &rSourceValues,
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

py::array_t<fp_t> Interpolate(
    const unsigned char execSpace, const method_t method,
    np_array<coordinates_t> sourcePoints, np_array<fp_t> sourceValues,
    optional_np_array<int_t> connVal, optional_np_array<offset_t> connOff,
    optional_np_array<cell_t> cellTypes, np_array<coordinates_t> targetPoints) {
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

  const int_t dim = sourcePoints.shape(1);
  switch (dim) {
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

} // namespace PyBindingsFiniteElements

} // namespace PACMAN
