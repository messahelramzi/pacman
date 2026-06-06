//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "common/types.hpp"

// clang-format off
#include "pybindings/bindings.hpp"
// clang-format on

namespace PACMAN {
namespace PyBindingsMLS {

/// @brief Function for MLS interpolation exposed to the Python interface.
/// @return A numpy array with the interpolated values at target points.
py::array_t<fp_t> Interpolate(const int_t spaceDimension,
                              const unsigned char execSpace,
                              np_array<coordinates_t> sourcePoints,
                              np_array<fp_t> sourceValues,
                              np_array<coordinates_t> targetPoints);

} // namespace PyBindingsMLS
} // namespace PACMAN
