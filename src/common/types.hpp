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
typedef int32_t geosupport_t;

enum class TransferMethods : method_t {
  NEAREST_NEAREST = 0xF0,
  INTERP_CLAMP = 0xF1,
  INTERP_NEAREST = 0xF2,
  INTERP_ZEROFILL = 0xF3,
  INTERP_EXTRAP = 0xF4,
  RBF_PUM = 0xF5,
};

enum class TransferStatus : status_t {
  OUTSIDE,
  NEAREST,
  INTER,
  EXTRAP,
  CLAMP,
};

enum class GeoSupport : geosupport_t {

  NA,
  POINT,
  LINE,
  TRIANGLE,
  QUAD,
  TETRA,
  HEXAHEDRON,
  WEDGE,
  PYRAMID,
};

enum class CellType : cell_t {

  // Isoparametric 0D
  VTK_EMPTY_CELL,
  VTK_VERTEX,
  // Isoparametric 1D
  VTK_LINE,
  VTK_QUADRATIC_EDGE,
  // Isoparametric 2D
  VTK_TRIANGLE,
  VTK_QUAD,
  VTK_QUADRATIC_TRIANGLE,
  VTK_QUADRATIC_QUAD,
  // Isoparametric 3D
  VTK_TETRA,
  VTK_HEXAHEDRON,
  VTK_WEDGE,
  VTK_PYRAMID,
  VTK_QUADRATIC_TETRA,
  VTK_QUADRATIC_HEXAHEDRON,
  VTK_QUADRATIC_WEDGE,
  VTK_QUADRATIC_PYRAMID,
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
