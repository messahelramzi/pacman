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
::std::tuple<np_array<fp_t>, np_array<int_t>>
Interpolate(const int_t spaceDimension, const unsigned char execSpace,
            const method_t method, np_array<coordinates_t> sourcePoints,
            np_array<fp_t> sourceValues, optional_np_array<int_t> connVal,
            optional_np_array<offset_t> connOff,
            optional_np_array<cell_t> cellTypes,
            np_array<coordinates_t> targetPoints);

int_t meshio_cell_dim(const ::std::string &cell_type);
int_t meshio_to_vtk_cell_type(const ::std::string &cell_type);

} // namespace PyBindingsFiniteElements
} // namespace PACMAN
