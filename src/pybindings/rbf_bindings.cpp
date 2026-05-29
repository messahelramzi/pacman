//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#include "pybindings/rbf_bindings.hpp"

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "pybindings/bindings.hpp"
// Only the class declaration is needed here — method bodies are provided by the
// ETI translation units (rbf_eti_*.cpp).  Extern template declarations in
// interpolator.hxx suppress re-instantiation in this TU.
#include "rbf_pum/interpolator.hxx"

namespace PACMAN {
namespace PyBindingsRbf {
template <int_t Dim, typename ArrayType>
inline auto GetPointAt(const ArrayType &rDataArray, const int_t i) {
  auto p = ::ArborX::Point<Dim, coordinates_t>{};
  for (int_t d = 0; d < Dim; ++d) {
    p[d] = rDataArray(i, d);
  }
  return p;
}

template <typename ExecSpace, typename RbfFuncType, int_t Dim>
py::array_t<fp_t> RunInterpolate(np_array<coordinates_t> &rSourcePoints,
                                 np_array<fp_t> &rSourceValues,
                                 np_array<coordinates_t> &rTargetPoints) {
  auto *p_source_points =
      static_cast<coordinates_t *>(rSourcePoints.mutable_data());
  auto *p_source_values = static_cast<fp_t *>(rSourceValues.mutable_data());
  auto *p_target_points =
      static_cast<coordinates_t *>(rTargetPoints.mutable_data());

  Transfer<ExecSpace, Dim> transfer(TransferMethods::RBF_PUM);
  SetupTransferClass(transfer, rSourcePoints.shape(0), 0, 0,
                     rTargetPoints.shape(0), p_source_points, p_source_values,
                     nullptr, nullptr, nullptr, p_target_points);
  auto rbf_interpolator =
      RbfPum::RbfPumInterpolator<ExecSpace, Dim, RbfFuncType>(transfer);

  auto target_values_np_array = py::array_t<fp_t>(rTargetPoints.shape(0));
  auto *p_target_values = target_values_np_array.mutable_data();
  auto unmanaged_host_tv =
      Kokkos::View<fp_t *, Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          p_target_values, rTargetPoints.shape(0));
  Kokkos::deep_copy(unmanaged_host_tv, transfer.targetValues);
  return target_values_np_array;
}

/// @brief This function allows us to pass runtime parameters as template
/// arguments, using a type union `std::variant` (`AvailableExecSpaces` &
/// `AvailableRbfFunctions`) to restraint types
/// @tparam Dim The space dimension of the problem, here the dimension is
/// checked to be between 1 and 3 (included)
template <int_t Dim>
requires IsValidDim<Dim> py::array_t<fp_t>
Dispatch(np_array<coordinates_t> &rSourcePoints, np_array<fp_t> &rSourceValues,
         np_array<coordinates_t> &rTargetPoints, const unsigned char execSpace,
         const unsigned char rbfFunction) {
  return std::visit(
      [&](auto execSpaceObj, auto rbfFunctionObj) {
        return RunInterpolate<decltype(execSpaceObj), decltype(rbfFunctionObj),
                              Dim>(rSourcePoints, rSourceValues, rTargetPoints);
      },
      MakeExecSpace(execSpace), MakeRbfFunction(rbfFunction));
}

py::array_t<fp_t> Interpolate(const int_t spaceDimension,
                              const unsigned char execSpace,
                              const unsigned char rbfFunction,
                              np_array<coordinates_t> sourcePoints,
                              np_array<fp_t> sourceValues,
                              np_array<coordinates_t> targetPoints) {
  const std::string _region_name = "PyBindingsRbf::Interpolate";
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
    return Dispatch<1>(sourcePoints, sourceValues, targetPoints, execSpace,
                       rbfFunction);
  case 2:
    return Dispatch<2>(sourcePoints, sourceValues, targetPoints, execSpace,
                       rbfFunction);
  case 3:
    return Dispatch<3>(sourcePoints, sourceValues, targetPoints, execSpace,
                       rbfFunction);
  default:
    throw new std::runtime_error(
        "The dimension of the points can only be: 1, 2 or 3.\n");
  }
}
} // namespace PyBindingsRbf
} // namespace PACMAN
