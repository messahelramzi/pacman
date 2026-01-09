#pragma once

#include <KokkosKernels_ArithTraits.hpp>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <iomanip>
#include <limits>

namespace PACMAN {
typedef double fp_t;
typedef double coordinates_t;

typedef int32_t int_t;
typedef int32_t shortint_t;
typedef uint32_t index_t;
typedef int32_t offset_t;

// types optionnels pour expliciter le type sous jacent des enums
typedef int32_t method_t;
typedef int32_t cell_t;
typedef int32_t status_t;

enum class TransferMethods : method_t {
  NEAREST_NEAREST = 0xF0,
  INTERP_CLAMP = 0xF1,
  INTERP_NEAREST = 0xF2,
  INTERP_ZEROFILL = 0xF3,
  INTERP_EXTRAP = 0xF4,
  RBF_PUM = 0xF5,
};

enum class TransferStatus : status_t {
  UNDEFINED = 0,
  NEAREST = 1,
  INTER = 2,
  EXTRAP = 3,
  CLAMP = 4,
  ZERO_FILL = 5,
  OUTSIDE = 6, // temporary
};

enum class CellType : cell_t {
  // Isoparametric linear cells using VTK numbering
  VTK_EMPTY_CELL = 0,
  VTK_VERTEX = 1,
  VTK_LINE = 3,
  VTK_TRIANGLE = 5,
  VTK_QUAD = 9,
  VTK_TETRA = 10,
  VTK_HEXAHEDRON = 12,
  VTK_WEDGE = 13,
  VTK_PYRAMID = 14,
  // Isoparametric quadratic cells using VTK numbering
  VTK_QUADRATIC_EDGE = 21,
  VTK_QUADRATIC_TRIANGLE = 22,
  VTK_QUADRATIC_QUAD = 23,
  VTK_QUADRATIC_TETRA = 24,
  VTK_QUADRATIC_HEXAHEDRON = 25,
  VTK_QUADRATIC_WEDGE = 26,
  VTK_QUADRATIC_PYRAMID = 27,
};

namespace fp_consts {
KOKKOS_INLINE_FUNCTION fp_t min(void) {
  return KokkosKernels::ArithTraits<fp_t>::min();
}

KOKKOS_INLINE_FUNCTION fp_t max(void) {
  return KokkosKernels::ArithTraits<fp_t>::max();
}
KOKKOS_INLINE_FUNCTION constexpr fp_t zero(void) {
  return static_cast<fp_t>(0.0);
}
KOKKOS_INLINE_FUNCTION constexpr fp_t one(void) {
  return static_cast<fp_t>(1.0);
}
KOKKOS_INLINE_FUNCTION constexpr fp_t epsilon(void) { return 1.0e-14; }
constexpr unsigned int precision = std::numeric_limits<fp_t>::max_digits10;
inline auto set_precision(void) {
  return std::setprecision(fp_consts::precision);
}
}; // namespace fp_consts

} // namespace PACMAN
