//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

// clang-format off
#include "pybindings/bindings.hpp"
// clang-format on

#include "common/transfer.hxx"
#include "common/types.hpp"

/// @brief Macro used to set the FE methods values into a Python submodule
#define ADD_FE_METHODS_ENTRY(M)                                                \
  fe_methods.attr(#M) = static_cast<method_t>(TransferMethods::M)

namespace PACMAN {
namespace PyBindingsFiniteElements {

/// @brief Interpolate function for FE methods exposed to the Python interface.
/// @return A tuple which contains the interpolated values and the status of the
/// interpolation for each point.
::std::tuple<np_array<fp_t>, np_array<int_t>>
Interpolate(const int_t spaceDimension, const unsigned char execSpace,
            const method_t method, np_array<coordinates_t> sourcePoints,
            np_array<fp_t> sourceValues, optional_np_array<int_t> connVal,
            optional_np_array<offset_t> connOff,
            optional_np_array<cell_t> cellTypes,
            np_array<coordinates_t> targetPoints);

np_array<int_t> vtk_to_pacman_cell_type(const np_array<int_t> cell_type);
int_t vtk_cell_dim(const int_t &cell_type);

} // namespace PyBindingsFiniteElements
} // namespace PACMAN
