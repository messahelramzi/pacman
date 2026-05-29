//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

// clang-format off
#include "pybindings/bindings.hpp"
// clang-format on

#include "solver/rbf_functions.hpp"

#define ADD_RBF_FUNCTIONS_ENTRY(F)                                             \
  rbf_functions.attr(#F) = PyBindingsRbf::RbfFunctions::F

namespace PACMAN {
namespace PyBindingsRbf {
using AvailableRbfFunctions =
    std::variant<RbfPum::WendlandC0, RbfPum::WendlandC2, RbfPum::WendlandC4,
                 RbfPum::WendlandC6, RbfPum::WendlandC8>;

/// @brief A struct which represents the available RBF functions using integer
/// values to receive it easily from Python
/// @note Struct entries underlying integer values are meaningless and can be
/// changed
struct RbfFunctions {
  static constexpr const unsigned char WENDLANDC0 = 0x10;
  static constexpr const unsigned char WENDLANDC2 = 0x11;
  static constexpr const unsigned char WENDLANDC4 = 0x12;
  static constexpr const unsigned char WENDLANDC6 = 0x13;
  static constexpr const unsigned char WENDLANDC8 = 0x14;
};

/// @brief Converts a RBF function from an unsigned char to a variant type
static inline AvailableRbfFunctions
MakeRbfFunction(const unsigned char rbfFunction) {
  switch (rbfFunction) {
  case RbfFunctions::WENDLANDC0:
    return RbfPum::WendlandC0{};
  case RbfFunctions::WENDLANDC2:
    return RbfPum::WendlandC2{};
  case RbfFunctions::WENDLANDC4:
    return RbfPum::WendlandC4{};
  case RbfFunctions::WENDLANDC6:
    return RbfPum::WendlandC6{};
  case RbfFunctions::WENDLANDC8:
    return RbfPum::WendlandC8{};
  default:
    std::cerr << "function: WendlandC0, WendlandC2, WendlandC4, "
                 "WendlandC6, WendlandC8"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

/// @brief Function for RBF inteprolation exposed to the Python interface.
/// @return A numpy array with the interpolated values at target points.
py::array_t<fp_t> Interpolate(const int_t spaceDimension,
                              const unsigned char execSpace,
                              const unsigned char rbfFunction,
                              np_array<coordinates_t> sourcePoints,
                              np_array<fp_t> sourceValues,
                              np_array<coordinates_t> targetPoints);
} // namespace PyBindingsRbf
} // namespace PACMAN
