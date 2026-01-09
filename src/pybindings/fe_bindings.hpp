#pragma once

// clang-format off
#include "pybindings/bindings.hpp"
// clang-format on

#include "common/transfer.hxx"
#include "common/types.hpp"

#define ADD_FE_METHODS_ENTRY(M)                                                \
  fe_methods.attr(#M) = static_cast<method_t>(TransferMethods::M)

namespace PACMAN {
namespace PyBindingsFiniteElements {
py::array_t<fp_t> Interpolate(
    const unsigned char execSpace, const method_t method,
    np_array<coordinates_t> sourcePoints, np_array<fp_t> sourceValues,
    optional_np_array<int_t> connVal, optional_np_array<offset_t> connOff,
    optional_np_array<cell_t> cellTypes, np_array<coordinates_t> targetPoints);
} // namespace PyBindingsFiniteElements
} // namespace PACMAN
