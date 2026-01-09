#pragma once

#include <Kokkos_Core.hpp>

#include "common/transfer.hxx"

namespace PACMAN {
namespace FiniteElements {

template <typename ExecSpace, CellType CT> struct LagrangeSpaceGeo;

template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_EMPTY_CELL> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 0U;
  static constexpr unsigned int dimensionality = 0;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    return;
  }
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t *, ExecSpace> sfv) {
    return;
  }
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    return;
  }
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_VERTEX> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 1U;
  static constexpr unsigned int dimensionality = 0;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    sfv(0) = 1.0;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_LINE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 2U;
  static constexpr unsigned int dimensionality = 1U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    sfv(0) = 1.0 - xi;
    sfv(1) = xi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    sfdv(0, 0) = -1.0;
    sfdv(0, 1) = 1;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_EDGE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 3U;
  static constexpr unsigned int dimensionality = 1U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 2.0 * xi - 1.0;
    sfv(0) = -x0 * (1.0 - xi);
    sfv(1) = x0 * xi;
    sfv(2) = xi * (4.0 - 4.0 * xi);
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 4.0 * xi;
    sfdv(0, 0) = x0 - 3.0;
    sfdv(0, 1) = x0 - 1.0;
    sfdv(0, 2) = 4.0 - 8.0 * xi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      sfddv(0, 0) = 4.0;
      break;
    }
    case 1: {
      sfddv(0, 0) = 4.0;
      break;
    }
    case 2: {
      sfddv(0, 0) = -8.0;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_TRIANGLE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 3U;
  static constexpr unsigned int dimensionality = 2U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    sfv(0) = -eta - xi + 1.0;
    sfv(1) = xi;
    sfv(2) = eta;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    sfdv(0, 0) = -1.0;
    sfdv(1, 0) = -1.0;
    sfdv(0, 1) = 1.0;
    sfdv(1, 1) = 0.0;
    sfdv(0, 2) = 0.0;
    sfdv(1, 2) = 1.0;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_TRIANGLE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 6U;
  static constexpr unsigned int dimensionality = 2U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 2 * eta;
    const fp_t x1 = 2 * xi - 1;
    const fp_t x2 = -eta - xi + 1;
    const fp_t x3 = 4 * xi;
    sfv(0) = x2 * (-x0 - x1);
    sfv(1) = x1 * xi;
    sfv(2) = eta * (x0 - 1.0);
    sfv(3) = x2 * x3;
    sfv(4) = eta * x3;
    sfv(5) = 4.0 * eta * x2;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 4.0 * eta;
    const fp_t x1 = 4.0 * xi;
    const fp_t x2 = x0 + x1 - 3.0;
    sfdv(0, 0) = x2;
    sfdv(1, 0) = x2;
    sfdv(0, 1) = x1 - 1.0;
    sfdv(1, 1) = 0.0;
    sfdv(0, 2) = 0.0;
    sfdv(1, 2) = x0 - 1.0;
    sfdv(0, 3) = -x0 - 8.0 * xi + 4.0;
    sfdv(1, 3) = -x1;
    sfdv(0, 4) = x0;
    sfdv(1, 4) = x1;
    sfdv(0, 5) = -x0;
    sfdv(1, 5) = -8.0 * eta - x1 + 4.0;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      sfddv(0, 0) = 4.0;
      sfddv(0, 1) = 4.0;
      sfddv(1, 0) = 4.0;
      sfddv(1, 1) = 4.0;
      break;
    }
    case 1: {
      sfddv(0, 0) = 4.0;
      break;
    }
    case 2: {
      sfddv(1, 1) = 4.0;
      break;
    }
    case 3: {
      sfddv(0, 0) = -8.0;
      sfddv(0, 1) = -4.0;
      sfddv(1, 0) = -4.0;
      break;
    }
    case 4: {
      sfddv(0, 1) = 4.0;
      sfddv(1, 0) = 4.0;
      break;
    }
    case 5: {
      sfddv(0, 1) = -4.0;
      sfddv(1, 0) = -4.0;
      sfddv(1, 1) = -8.0;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUAD> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 4U;
  static constexpr unsigned int dimensionality = 2U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 1 - eta;
    const fp_t x1 = 1 - xi;
    sfv(0) = x0 * x1;
    sfv(1) = x0 * xi;
    sfv(2) = eta * xi;
    sfv(3) = eta * x1;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = eta - 1;
    const fp_t x1 = xi - 1;
    sfdv(0, 0) = x0;
    sfdv(1, 0) = x1;
    sfdv(0, 1) = -x0;
    sfdv(1, 1) = -xi;
    sfdv(0, 2) = eta;
    sfdv(1, 2) = xi;
    sfdv(0, 3) = -eta;
    sfdv(1, 3) = -x1;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      sfddv(0, 1) = 1;
      sfddv(1, 0) = 1;
      break;
    }
    case 1: {
      sfddv(0, 1) = -1;
      sfddv(1, 0) = -1;
      break;
    }
    case 2: {
      sfddv(0, 1) = 1;
      sfddv(1, 0) = 1;
      break;
    }
    case 3: {
      sfddv(0, 1) = -1;
      sfddv(1, 0) = -1;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_QUAD> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 8U;
  static constexpr unsigned int dimensionality = 2U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 1.0 - eta;
    const fp_t x1 = 1.0 - xi;
    const fp_t x2 = 2.0 * eta;
    const fp_t x3 = 2.0 * xi;
    const fp_t x4 = x3 - 1.0;
    const fp_t x5 = 1.0 - x4 * x4;
    const fp_t x6 = 1.0 - (x2 - 1.0) * (x2 - 1.0);
    sfv(0) = x0 * x1 * (-x2 - x4);
    sfv(1) = x0 * xi * (-x2 + x3 - 1);
    sfv(2) = eta * xi * (x2 + x3 - 3);
    sfv(3) = eta * x1 * (x2 - x3 - 1);
    sfv(4) = 0.5 * x5 * (2.0 - x2);
    sfv(5) = x6 * xi;
    sfv(6) = eta * x5;
    sfv(7) = 0.5 * x6 * (2.0 - x3);
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 1.0 - eta;
    const fp_t x1 = 1.0 - xi;
    const fp_t x2 = 2.0 * x0 * x1;
    const fp_t x3 = 2.0 * xi;
    const fp_t x4 = 2.0 * eta;
    const fp_t x5 = x4 - 1.0;
    const fp_t x6 = -x3 - x5;
    const fp_t x7 = x0 * x3;
    const fp_t x8 = x3 - x4 - 1.0;
    const fp_t x9 = x4 * xi;
    const fp_t x10 = x3 + x4 - 3.0;
    const fp_t x11 = x1 * x4;
    const fp_t x12 = -x3 + x4 - 1.0;
    const fp_t x13 = 4.0 - 8.0 * xi;
    const fp_t x14 = x5 * x5 - 1;
    const fp_t x15 = (x3 - 1.0) * (x3 - 1.0) - 1.0;
    const fp_t x16 = 4.0 - 8.0 * eta;
    sfdv(0, 0) = -x0 * x6 - x2;
    sfdv(1, 0) = -x1 * x6 - x2;
    sfdv(0, 1) = x0 * x8 + x7;
    sfdv(1, 1) = -x7 - x8 * xi;
    sfdv(0, 2) = eta * x10 + x9;
    sfdv(1, 2) = x10 * xi + x9;
    sfdv(0, 3) = -eta * x12 - x11;
    sfdv(1, 3) = x1 * x12 + x11;
    sfdv(0, 4) = 0.5 * x13 * (2.0 - x4);
    sfdv(1, 4) = x15;
    sfdv(0, 5) = -x14;
    sfdv(1, 5) = x16 * xi;
    sfdv(0, 6) = eta * x13;
    sfdv(1, 6) = -x15;
    sfdv(0, 7) = x14;
    sfdv(1, 7) = 0.5 * x16 * (2.0 - x3);
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = 4 * eta;
      const fp_t x1 = 4 * xi;
      const fp_t x2 = -x0 - x1 + 5;
      sfddv(0, 0) = 4 - x0;
      sfddv(0, 1) = x2;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = 4 - x1;
      break;
    }
    case 1: {
      const fp_t x0 = 4 * eta;
      const fp_t x1 = 4 * xi;
      const fp_t x2 = x0 - x1 - 1;
      sfddv(0, 0) = 4 - x0;
      sfddv(0, 1) = x2;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = x1;
      break;
    }
    case 2: {
      const fp_t x0 = 4 * eta;
      const fp_t x1 = 4 * xi;
      const fp_t x2 = x0 + x1 - 3;
      sfddv(0, 0) = x0;
      sfddv(0, 1) = x2;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = x1;
      break;
    }
    case 3: {
      const fp_t x0 = 4 * eta;
      const fp_t x1 = 4 * xi;
      const fp_t x2 = -x0 + x1 - 1;
      sfddv(0, 0) = x0;
      sfddv(0, 1) = x2;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = 4 - x1;
      break;
    }
    case 4: {
      const fp_t x0 = 8 * xi - 4;
      sfddv(0, 0) = 8 * eta - 8;
      sfddv(0, 1) = x0;
      sfddv(1, 0) = x0;
      break;
    }
    case 5: {
      const fp_t x0 = 4 - 8 * eta;
      sfddv(0, 1) = x0;
      sfddv(1, 0) = x0;
      sfddv(1, 1) = -8 * xi;
      break;
    }
    case 6: {
      const fp_t x0 = 4 - 8 * xi;
      sfddv(0, 0) = -8 * eta;
      sfddv(0, 1) = x0;
      sfddv(1, 0) = x0;
      break;
    }
    case 7: {
      const fp_t x0 = 8 * eta - 4;
      sfddv(0, 1) = x0;
      sfddv(1, 0) = x0;
      sfddv(1, 1) = 8 * xi - 8;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_TETRA> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 4U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    sfv(0) = -eta - phi - xi + 1.0;
    sfv(1) = xi;
    sfv(2) = eta;
    sfv(3) = phi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    sfdv(0, 0) = -1.0;
    sfdv(1, 0) = -1.0;
    sfdv(2, 0) = -1.0;
    sfdv(0, 1) = 1.0;
    sfdv(1, 1) = 0.0;
    sfdv(2, 1) = 0.0;
    sfdv(0, 2) = 0.0;
    sfdv(1, 2) = 1.0;
    sfdv(2, 2) = 0.0;
    sfdv(0, 3) = 0.0;
    sfdv(1, 3) = 0.0;
    sfdv(2, 3) = 1.0;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_TETRA> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 10U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 2.0 * eta - 1.0;
    const fp_t x1 = 2.0 * xi;
    const fp_t x2 = 2.0 * phi;
    const fp_t x3 = -eta - phi - xi + 1.0;
    const fp_t x4 = 4.0 * eta;
    const fp_t x5 = 4.0 * phi;
    const fp_t x6 = -x4 - x5 - 4.0 * xi + 4.0;
    sfv(0) = eta * x0;
    sfv(1) = x3 * (-x0 - x1 - x2);
    sfv(2) = xi * (x1 - 1.0);
    sfv(3) = phi * (x2 - 1.0);
    sfv(4) = x3 * x4;
    sfv(5) = x6 * xi;
    sfv(6) = x4 * xi;
    sfv(7) = phi * x4;
    sfv(8) = phi * x6;
    sfv(9) = x5 * xi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 4.0 * xi;
    const fp_t x1 = 4.0 * eta;
    const fp_t x2 = 4.0 * phi;
    const fp_t x3 = x1 + x2;
    const fp_t x4 = x0 + x3 - 3.0;
    const fp_t x5 = -x1;
    const fp_t x6 = -x2;
    const fp_t x7 = x0 - 4.0;
    const fp_t x8 = -x0;
    sfdv(0, 0) = 0.0;
    sfdv(1, 0) = x1 - 1.0;
    sfdv(2, 0) = 0.0;
    sfdv(0, 1) = x4;
    sfdv(1, 1) = x4;
    sfdv(2, 1) = x4;
    sfdv(0, 2) = x0 - 1.0;
    sfdv(1, 2) = 0.0;
    sfdv(2, 2) = 0.0;
    sfdv(0, 3) = 0.0;
    sfdv(1, 3) = 0.0;
    sfdv(2, 3) = x2 - 1.0;
    sfdv(0, 4) = x5;
    sfdv(1, 4) = -8.0 * eta - x2 - x7;
    sfdv(2, 4) = x5;
    sfdv(0, 5) = -x3 - 8.0 * xi + 4.0;
    sfdv(1, 5) = x8;
    sfdv(2, 5) = x8;
    sfdv(0, 6) = x1;
    sfdv(1, 6) = x0;
    sfdv(2, 6) = 0.0;
    sfdv(0, 7) = 0.0;
    sfdv(1, 7) = x2;
    sfdv(2, 7) = x1;
    sfdv(0, 8) = x6;
    sfdv(1, 8) = x6;
    sfdv(2, 8) = -8.0 * phi - x1 - x7;
    sfdv(0, 9) = x2;
    sfdv(1, 9) = 0.0;
    sfdv(2, 9) = x0;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      sfddv(1, 1) = 4.0;
      break;
    }
    case 1: {
      sfddv(0, 0) = 4.0;
      sfddv(0, 1) = 4.0;
      sfddv(0, 2) = 4.0;
      sfddv(1, 0) = 4.0;
      sfddv(1, 1) = 4.0;
      sfddv(1, 2) = 4.0;
      sfddv(2, 0) = 4.0;
      sfddv(2, 1) = 4.0;
      sfddv(2, 2) = 4.0;
      break;
    }
    case 2: {
      sfddv(0, 0) = 4.0;
      break;
    }
    case 3: {
      sfddv(2, 2) = 4.0;
      break;
    }
    case 4: {
      sfddv(0, 1) = -4.0;
      sfddv(1, 0) = -4.0;
      sfddv(1, 1) = -8.0;
      sfddv(1, 2) = -4.0;
      sfddv(2, 1) = -4.0;
      break;
    }
    case 5: {
      sfddv(0, 0) = -8.0;
      sfddv(0, 1) = -4.0;
      sfddv(0, 2) = -4.0;
      sfddv(1, 0) = -4.0;
      sfddv(2, 0) = -4.0;
      break;
    }
    case 6: {
      sfddv(0, 1) = 4.0;
      sfddv(1, 0) = 4.0;
      break;
    }
    case 7: {
      sfddv(1, 2) = 4.0;
      sfddv(2, 1) = 4.0;
      break;
    }
    case 8: {
      sfddv(0, 2) = -4.0;
      sfddv(1, 2) = -4.0;
      sfddv(2, 0) = -4.0;
      sfddv(2, 1) = -4.0;
      sfddv(2, 2) = -8.0;
      break;
    }
    case 9: {
      sfddv(0, 2) = 4.0;
      sfddv(2, 0) = 4.0;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_HEXAHEDRON> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 8U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 1 - xi;
    const fp_t x1 = 1 - eta;
    const fp_t x2 = 1 - phi;
    const fp_t x3 = x1 * x2;
    const fp_t x4 = eta * x2;
    const fp_t x5 = phi * x1;
    const fp_t x6 = eta * phi;
    sfv(0) = x0 * x3;
    sfv(1) = x3 * xi;
    sfv(2) = x4 * xi;
    sfv(3) = x0 * x4;
    sfv(4) = x0 * x5;
    sfv(5) = x5 * xi;
    sfv(6) = x6 * xi;
    sfv(7) = x0 * x6;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 1 - eta;
    const fp_t x1 = 1 - phi;
    const fp_t x2 = x0 * x1;
    const fp_t x3 = eta * x1;
    const fp_t x4 = phi * x0;
    const fp_t x5 = eta * phi;
    const fp_t x6 = 1 - xi;
    const fp_t x7 = x1 * x6;
    const fp_t x8 = x1 * xi;
    const fp_t x9 = phi * x6;
    const fp_t x10 = phi * xi;
    const fp_t x11 = x0 * x6;
    const fp_t x12 = x0 * xi;
    const fp_t x13 = eta * xi;
    const fp_t x14 = eta * x6;
    sfdv(0, 0) = -x2;
    sfdv(1, 0) = -x7;
    sfdv(2, 0) = -x11;
    sfdv(0, 1) = x2;
    sfdv(1, 1) = -x8;
    sfdv(2, 1) = -x12;
    sfdv(0, 2) = x3;
    sfdv(1, 2) = x8;
    sfdv(2, 2) = -x13;
    sfdv(0, 3) = -x3;
    sfdv(1, 3) = x7;
    sfdv(2, 3) = -x14;
    sfdv(0, 4) = -x4;
    sfdv(1, 4) = -x9;
    sfdv(2, 4) = x11;
    sfdv(0, 5) = x4;
    sfdv(1, 5) = -x10;
    sfdv(2, 5) = x12;
    sfdv(0, 6) = x5;
    sfdv(1, 6) = x10;
    sfdv(2, 6) = x13;
    sfdv(0, 7) = -x5;
    sfdv(1, 7) = x9;
    sfdv(2, 7) = x14;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 1 - eta;
      const fp_t x2 = 1 - xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 1: {
      const fp_t x0 = phi - 1;
      const fp_t x1 = eta - 1;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = xi;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = xi;
      break;
    }
    case 2: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = -eta;
      const fp_t x2 = -xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 3: {
      const fp_t x0 = phi - 1;
      const fp_t x1 = xi - 1;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = eta;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x1;
      sfddv(2, 0) = eta;
      sfddv(2, 1) = x1;
      break;
    }
    case 4: {
      const fp_t x0 = eta - 1;
      const fp_t x1 = xi - 1;
      sfddv(0, 1) = phi;
      sfddv(0, 2) = x0;
      sfddv(1, 0) = phi;
      sfddv(1, 2) = x1;
      sfddv(2, 0) = x0;
      sfddv(2, 1) = x1;
      break;
    }
    case 5: {
      const fp_t x0 = -phi;
      const fp_t x1 = 1 - eta;
      const fp_t x2 = -xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 6: {
      sfddv(0, 1) = phi;
      sfddv(0, 2) = eta;
      sfddv(1, 0) = phi;
      sfddv(1, 2) = xi;
      sfddv(2, 0) = eta;
      sfddv(2, 1) = xi;
      break;
    }
    case 7: {
      const fp_t x0 = -phi;
      const fp_t x1 = -eta;
      const fp_t x2 = 1 - xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_HEXAHEDRON> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 20U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 1 - xi;
    const fp_t x1 = 2 * xi;
    const fp_t x2 = 2 * eta;
    const fp_t x3 = 2 * phi;
    const fp_t x4 = x2 + x3;
    const fp_t x5 = x1 + x4;
    const fp_t x6 = 1 - eta;
    const fp_t x7 = 1 - phi;
    const fp_t x8 = x6 * x7;
    const fp_t x9 = -x1;
    const fp_t x10 = x4 + x9;
    const fp_t x11 = x2 - x3;
    const fp_t x12 = x1 + x11;
    const fp_t x13 = eta * x7;
    const fp_t x14 = x13 * xi;
    const fp_t x15 = phi * x6;
    const fp_t x16 = x15 * xi;
    const fp_t x17 = eta * phi;
    const fp_t x18 = x17 * xi;
    const fp_t x19 = 2 - x2;
    const fp_t x20 = 2 - x3;
    const fp_t x21 = x20 * xi;
    const fp_t x22 = x0 * x21;
    const fp_t x23 = 2 - x1;
    const fp_t x24 = x3 * xi;
    const fp_t x25 = x19 * x7;
    const fp_t x26 = phi * x23;
    sfv(0) = x0 * x8 * (1 - x5);
    sfv(1) = x8 * xi * (-x10 - 1);
    sfv(2) = x14 * (x12 - 3);
    sfv(3) = x0 * x13 * (-x1 + x2 - x3 - 1);
    sfv(4) = x0 * x15 * (-x12 - 1);
    sfv(5) = x16 * (-x11 - x9 - 3);
    sfv(6) = x18 * (x5 - 5);
    sfv(7) = x0 * x17 * (x10 - 3);
    sfv(8) = x19 * x22;
    sfv(9) = x2 * x21 * x6;
    sfv(10) = x2 * x22;
    sfv(11) = eta * x20 * x23 * x6;
    sfv(12) = x0 * x19 * x24;
    sfv(13) = 4 * eta * x16;
    sfv(14) = 4 * x0 * x18;
    sfv(15) = x15 * x2 * x23;
    sfv(16) = x25 * x26;
    sfv(17) = x24 * x25;
    sfv(18) = 4 * phi * x14;
    sfv(19) = x2 * x26 * x7;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 1 - xi;
    const fp_t x1 = 1 - eta;
    const fp_t x2 = 1 - phi;
    const fp_t x3 = x1 * x2;
    const fp_t x4 = 2 * x0 * x3;
    const fp_t x5 = 2 * xi;
    const fp_t x6 = 2 * eta;
    const fp_t x7 = 2 * phi;
    const fp_t x8 = x6 + x7;
    const fp_t x9 = x5 + x8;
    const fp_t x10 = 1 - x9;
    const fp_t x11 = x3 * x5;
    const fp_t x12 = -x5;
    const fp_t x13 = x12 + x8;
    const fp_t x14 = -x13 - 1;
    const fp_t x15 = x2 * xi;
    const fp_t x16 = x15 * x6;
    const fp_t x17 = x6 - x7;
    const fp_t x18 = x17 + x5;
    const fp_t x19 = x18 - 3;
    const fp_t x20 = eta * x2;
    const fp_t x21 = x0 * x2;
    const fp_t x22 = x21 * x6;
    const fp_t x23 = -x5 + x6 - x7 - 1;
    const fp_t x24 = x0 * x1;
    const fp_t x25 = x24 * x7;
    const fp_t x26 = -x18 - 1;
    const fp_t x27 = phi * x1;
    const fp_t x28 = x1 * xi;
    const fp_t x29 = x28 * x7;
    const fp_t x30 = -x12 - x17 - 3;
    const fp_t x31 = phi * xi;
    const fp_t x32 = x31 * x6;
    const fp_t x33 = x9 - 5;
    const fp_t x34 = eta * phi;
    const fp_t x35 = phi * x0;
    const fp_t x36 = x35 * x6;
    const fp_t x37 = x13 - 3;
    const fp_t x38 = 2 - x6;
    const fp_t x39 = 2 - x7;
    const fp_t x40 = x39 * x6;
    const fp_t x41 = x1 * x40;
    const fp_t x42 = x40 * xi;
    const fp_t x43 = x38 * x7;
    const fp_t x44 = x43 * xi;
    const fp_t x45 = 4 * eta * x27;
    const fp_t x46 = 4 * x34 * xi;
    const fp_t x47 = x2 * x43;
    const fp_t x48 = 4 * phi;
    const fp_t x49 = x20 * x48;
    const fp_t x50 = x0 * x39 * x5;
    const fp_t x51 = 2 - x5;
    const fp_t x52 = 4 * x35 * xi;
    const fp_t x53 = x51 * x6;
    const fp_t x54 = phi * x53;
    const fp_t x55 = x2 * x51 * x7;
    const fp_t x56 = x15 * x48;
    const fp_t x57 = eta * xi;
    const fp_t x58 = eta * x0;
    const fp_t x59 = x0 * x38 * x5;
    const fp_t x60 = 4 * eta * x28;
    const fp_t x61 = 4 * x0 * x57;
    const fp_t x62 = x1 * x53;
    sfdv(0, 0) = -x10 * x3 - x4;
    sfdv(1, 0) = -x10 * x21 - x4;
    sfdv(2, 0) = -x10 * x24 - x4;
    sfdv(0, 1) = x11 + x14 * x3;
    sfdv(1, 1) = -x11 - x14 * x15;
    sfdv(2, 1) = -x11 - x14 * x28;
    sfdv(0, 2) = x16 + x19 * x20;
    sfdv(1, 2) = x15 * x19 + x16;
    sfdv(2, 2) = -x16 - x19 * x57;
    sfdv(0, 3) = -x20 * x23 - x22;
    sfdv(1, 3) = x21 * x23 + x22;
    sfdv(2, 3) = -x22 - x23 * x58;
    sfdv(0, 4) = -x25 - x26 * x27;
    sfdv(1, 4) = -x25 - x26 * x35;
    sfdv(2, 4) = x24 * x26 + x25;
    sfdv(0, 5) = x27 * x30 + x29;
    sfdv(1, 5) = -x29 - x30 * x31;
    sfdv(2, 5) = x28 * x30 + x29;
    sfdv(0, 6) = x32 + x33 * x34;
    sfdv(1, 6) = x31 * x33 + x32;
    sfdv(2, 6) = x32 + x33 * x57;
    sfdv(0, 7) = -x34 * x37 - x36;
    sfdv(1, 7) = x35 * x37 + x36;
    sfdv(2, 7) = x36 + x37 * x58;
    sfdv(0, 8) = x0 * x38 * x39 - x38 * x39 * xi;
    sfdv(1, 8) = -x50;
    sfdv(2, 8) = -x59;
    sfdv(0, 9) = x41;
    sfdv(1, 9) = 2 * x1 * x39 * xi - x42;
    sfdv(2, 9) = -x60;
    sfdv(0, 10) = 2 * eta * x0 * x39 - x42;
    sfdv(1, 10) = x50;
    sfdv(2, 10) = -x61;
    sfdv(0, 11) = -x41;
    sfdv(1, 11) = -eta * x39 * x51 + x1 * x39 * x51;
    sfdv(2, 11) = -x62;
    sfdv(0, 12) = 2 * phi * x0 * x38 - x44;
    sfdv(1, 12) = -x52;
    sfdv(2, 12) = x59;
    sfdv(0, 13) = x45;
    sfdv(1, 13) = 4 * phi * x1 * xi - x46;
    sfdv(2, 13) = x60;
    sfdv(0, 14) = 4 * eta * phi * x0 - x46;
    sfdv(1, 14) = x52;
    sfdv(2, 14) = x61;
    sfdv(0, 15) = -x45;
    sfdv(1, 15) = 2 * phi * x1 * x51 - x54;
    sfdv(2, 15) = x62;
    sfdv(0, 16) = -x47;
    sfdv(1, 16) = -x55;
    sfdv(2, 16) = -phi * x38 * x51 + x2 * x38 * x51;
    sfdv(0, 17) = x47;
    sfdv(1, 17) = -x56;
    sfdv(2, 17) = 2 * x2 * x38 * xi - x44;
    sfdv(0, 18) = x49;
    sfdv(1, 18) = x56;
    sfdv(2, 18) = 4 * eta * x2 * xi - x46;
    sfdv(0, 19) = -x49;
    sfdv(1, 19) = x55;
    sfdv(2, 19) = 2 * eta * x2 * x51 - x54;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 1 - eta;
      const fp_t x2 = x0 * x1;
      const fp_t x3 = 2 * x2;
      const fp_t x4 = 1 - xi;
      const fp_t x5 = 2 * x4;
      const fp_t x6 = x0 * x5;
      const fp_t x7 = -2 * eta - 2 * phi - 2 * xi + 1;
      const fp_t x8 = x0 * x7 + x3 + x6;
      const fp_t x9 = x1 * x5;
      const fp_t x10 = x1 * x7 + x3 + x9;
      const fp_t x11 = 4 * x4;
      const fp_t x12 = x4 * x7 + x6 + x9;
      sfddv(0, 0) = 4 * x2;
      sfddv(0, 1) = x8;
      sfddv(0, 2) = x10;
      sfddv(1, 0) = x8;
      sfddv(1, 1) = x0 * x11;
      sfddv(1, 2) = x12;
      sfddv(2, 0) = x10;
      sfddv(2, 1) = x12;
      sfddv(2, 2) = x1 * x11;
      break;
    }
    case 1: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 1 - eta;
      const fp_t x2 = x0 * x1;
      const fp_t x3 = 2 * xi;
      const fp_t x4 = x0 * x3;
      const fp_t x5 = 2 * x2;
      const fp_t x6 = -2 * eta - 2 * phi + 2 * xi - 1;
      const fp_t x7 = -x0 * x6 - x4 - x5;
      const fp_t x8 = x1 * x3;
      const fp_t x9 = -x1 * x6 - x5 - x8;
      const fp_t x10 = 4 * xi;
      const fp_t x11 = x4 + x6 * xi + x8;
      sfddv(0, 0) = 4 * x2;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = x0 * x10;
      sfddv(1, 2) = x11;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x11;
      sfddv(2, 2) = x1 * x10;
      break;
    }
    case 2: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 4 * x0;
      const fp_t x2 = 2 * eta;
      const fp_t x3 = x0 * x2;
      const fp_t x4 = 2 * xi;
      const fp_t x5 = x0 * x4;
      const fp_t x6 = -2 * phi + x2 + x4 - 3;
      const fp_t x7 = x0 * x6 + x3 + x5;
      const fp_t x8 = x2 * xi;
      const fp_t x9 = -eta * x6 - x3 - x8;
      const fp_t x10 = -x5 - x6 * xi - x8;
      sfddv(0, 0) = eta * x1;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = x1 * xi;
      sfddv(1, 2) = x10;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x10;
      sfddv(2, 2) = 4 * eta * xi;
      break;
    }
    case 3: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 4 * x0;
      const fp_t x2 = 2 * eta;
      const fp_t x3 = x0 * x2;
      const fp_t x4 = 1 - xi;
      const fp_t x5 = 2 * x0 * x4;
      const fp_t x6 = 2 * eta - 2 * phi - 2 * xi - 1;
      const fp_t x7 = -x0 * x6 - x3 - x5;
      const fp_t x8 = x2 * x4;
      const fp_t x9 = eta * x6 + x3 + x8;
      const fp_t x10 = -x4 * x6 - x5 - x8;
      sfddv(0, 0) = eta * x1;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = x1 * x4;
      sfddv(1, 2) = x10;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x10;
      sfddv(2, 2) = 4 * eta * x4;
      break;
    }
    case 4: {
      const fp_t x0 = 1 - eta;
      const fp_t x1 = 4 * phi;
      const fp_t x2 = 2 * phi;
      const fp_t x3 = x0 * x2;
      const fp_t x4 = 1 - xi;
      const fp_t x5 = x2 * x4;
      const fp_t x6 = -2 * eta + 2 * phi - 2 * xi - 1;
      const fp_t x7 = phi * x6 + x3 + x5;
      const fp_t x8 = x0 * x4;
      const fp_t x9 = 2 * x8;
      const fp_t x10 = -x0 * x6 - x3 - x9;
      const fp_t x11 = -x4 * x6 - x5 - x9;
      sfddv(0, 0) = x0 * x1;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x10;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = x1 * x4;
      sfddv(1, 2) = x11;
      sfddv(2, 0) = x10;
      sfddv(2, 1) = x11;
      sfddv(2, 2) = 4 * x8;
      break;
    }
    case 5: {
      const fp_t x0 = 1 - eta;
      const fp_t x1 = 4 * phi;
      const fp_t x2 = 2 * phi;
      const fp_t x3 = x2 * xi;
      const fp_t x4 = x0 * x2;
      const fp_t x5 = -2 * eta + 2 * phi + 2 * xi - 3;
      const fp_t x6 = -phi * x5 - x3 - x4;
      const fp_t x7 = 2 * x0 * xi;
      const fp_t x8 = x0 * x5 + x4 + x7;
      const fp_t x9 = -x3 - x5 * xi - x7;
      sfddv(0, 0) = x0 * x1;
      sfddv(0, 1) = x6;
      sfddv(0, 2) = x8;
      sfddv(1, 0) = x6;
      sfddv(1, 1) = x1 * xi;
      sfddv(1, 2) = x9;
      sfddv(2, 0) = x8;
      sfddv(2, 1) = x9;
      sfddv(2, 2) = 4 * x0 * xi;
      break;
    }
    case 6: {
      const fp_t x0 = 4 * phi;
      const fp_t x1 = 2 * eta;
      const fp_t x2 = phi * x1;
      const fp_t x3 = 2 * phi;
      const fp_t x4 = x3 * xi;
      const fp_t x5 = x1 + x3 + 2 * xi - 5;
      const fp_t x6 = phi * x5 + x2 + x4;
      const fp_t x7 = x1 * xi;
      const fp_t x8 = eta * x5 + x2 + x7;
      const fp_t x9 = x4 + x5 * xi + x7;
      sfddv(0, 0) = eta * x0;
      sfddv(0, 1) = x6;
      sfddv(0, 2) = x8;
      sfddv(1, 0) = x6;
      sfddv(1, 1) = x0 * xi;
      sfddv(1, 2) = x9;
      sfddv(2, 0) = x8;
      sfddv(2, 1) = x9;
      sfddv(2, 2) = 4 * eta * xi;
      break;
    }
    case 7: {
      const fp_t x0 = 4 * phi;
      const fp_t x1 = 2 * eta;
      const fp_t x2 = phi * x1;
      const fp_t x3 = 1 - xi;
      const fp_t x4 = 2 * phi;
      const fp_t x5 = x3 * x4;
      const fp_t x6 = x1 + x4 - 2 * xi - 3;
      const fp_t x7 = -phi * x6 - x2 - x5;
      const fp_t x8 = x1 * x3;
      const fp_t x9 = -eta * x6 - x2 - x8;
      const fp_t x10 = x3 * x6 + x5 + x8;
      sfddv(0, 0) = eta * x0;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = x0 * x3;
      sfddv(1, 2) = x10;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x10;
      sfddv(2, 2) = 4 * eta * x3;
      break;
    }
    case 8: {
      const fp_t x0 = 2 - 2 * eta;
      const fp_t x1 = 4 - 4 * phi;
      const fp_t x2 = 1 - xi;
      const fp_t x3 = -x1 * x2 + x1 * xi;
      const fp_t x4 = 2 * x0;
      const fp_t x5 = -x2 * x4 + x4 * xi;
      const fp_t x6 = 4 * x2 * xi;
      sfddv(0, 0) = -x0 * x1;
      sfddv(0, 1) = x3;
      sfddv(0, 2) = x5;
      sfddv(1, 0) = x3;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x5;
      sfddv(2, 1) = x6;
      break;
    }
    case 9: {
      const fp_t x0 = 2.0 - 2.0 * phi;
      const fp_t x1 = 1.0 - eta;
      const fp_t x2 = -2.0 * eta * x0 + 2.0 * x0 * x1;
      const fp_t x3 = 4.0 * x1;
      const fp_t x4 = -eta * x3;
      const fp_t x5 = 4.0 * xi;
      const fp_t x6 = eta * x5 - x3 * xi;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = -x0 * x5;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x6;
      break;
    }
    case 10: {
      const fp_t x0 = 2.0 - 2.0 * phi;
      const fp_t x1 = 4.0 * eta;
      const fp_t x2 = 1.0 - xi;
      const fp_t x3 = 2.0 * x0 * x2 - 2.0 * x0 * xi;
      const fp_t x4 = -x1 * x2 + x1 * xi;
      const fp_t x5 = -4.0 * x2 * xi;
      sfddv(0, 0) = -x0 * x1;
      sfddv(0, 1) = x3;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x3;
      sfddv(1, 2) = x5;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x5;
      break;
    }
    case 11: {
      const fp_t x0 = 4.0 - 4.0 * phi;
      const fp_t x1 = 1.0 - eta;
      const fp_t x2 = eta * x0 - x0 * x1;
      const fp_t x3 = 4.0 * eta * x1;
      const fp_t x4 = 2.0 - 2.0 * xi;
      const fp_t x5 = 2.0 * x4;
      const fp_t x6 = eta * x5 - x1 * x5;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = -x0 * x4;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x6;
      break;
    }
    case 12: {
      const fp_t x0 = 2.0 - 2.0 * eta;
      const fp_t x1 = 4.0 * phi;
      const fp_t x2 = 1.0 - xi;
      const fp_t x3 = -x1 * x2 + x1 * xi;
      const fp_t x4 = 2.0 * x0 * x2 - 2.0 * x0 * xi;
      const fp_t x5 = -4.0 * x2 * xi;
      sfddv(0, 0) = -x0 * x1;
      sfddv(0, 1) = x3;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x3;
      sfddv(1, 2) = x5;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x5;
      break;
    }
    case 13: {
      const fp_t x0 = 1.0 - eta;
      const fp_t x1 = -4.0 * eta * phi + 4.0 * phi * x0;
      const fp_t x2 = 4.0 * eta;
      const fp_t x3 = x0 * x2;
      const fp_t x4 = 4.0 * x0 * xi - x2 * xi;
      sfddv(0, 1) = x1;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x1;
      sfddv(1, 1) = -8 * phi * xi;
      sfddv(1, 2) = x4;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x4;
      break;
    }
    case 14: {
      const fp_t x0 = 1.0 - xi;
      const fp_t x1 = 4.0 * phi * x0 - 4.0 * phi * xi;
      const fp_t x2 = 4.0 * eta * x0 - 4.0 * eta * xi;
      const fp_t x3 = 4.0 * x0 * xi;
      sfddv(0, 0) = -8 * eta * phi;
      sfddv(0, 1) = x1;
      sfddv(0, 2) = x2;
      sfddv(1, 0) = x1;
      sfddv(1, 2) = x3;
      sfddv(2, 0) = x2;
      sfddv(2, 1) = x3;
      break;
    }
    case 15: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 1.0 - eta;
      const fp_t x2 = eta * x0 - x0 * x1;
      const fp_t x3 = -4.0 * eta * x1;
      const fp_t x4 = 2.0 - 2.0 * xi;
      const fp_t x5 = -2.0 * eta * x4 + 2.0 * x1 * x4;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x2;
      sfddv(1, 1) = -x0 * x4;
      sfddv(1, 2) = x5;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x5;
      break;
    }
    case 16: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 4.0 * phi * x0;
      const fp_t x2 = 4.0 - 4.0 * eta;
      const fp_t x3 = phi * x2 - x0 * x2;
      const fp_t x4 = 2.0 - 2.0 * xi;
      const fp_t x5 = 2.0 * x4;
      const fp_t x6 = phi * x5 - x0 * x5;
      sfddv(0, 1) = x1;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x1;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -x2 * x4;
      break;
    }
    case 17: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 4.0 * x0;
      const fp_t x2 = -phi * x1;
      const fp_t x3 = 2.0 - 2.0 * eta;
      const fp_t x4 = -2.0 * phi * x3 + 2.0 * x0 * x3;
      const fp_t x5 = 4.0 * xi;
      const fp_t x6 = phi * x5 - x1 * xi;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -x3 * x5;
      break;
    }
    case 18: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 4.0 * phi;
      const fp_t x2 = x0 * x1;
      const fp_t x3 = 4.0 * eta * x0 - eta * x1;
      const fp_t x4 = 4.0 * x0 * xi - x1 * xi;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x4;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x4;
      sfddv(2, 2) = -8.0 * eta * xi;
      break;
    }
    case 19: {
      const fp_t x0 = 1 - phi;
      const fp_t x1 = 4.0 * x0;
      const fp_t x2 = -phi * x1;
      const fp_t x3 = 4.0 * eta;
      const fp_t x4 = -eta * x1 + phi * x3;
      const fp_t x5 = 2.0 - 2.0 * xi;
      const fp_t x6 = -2.0 * phi * x5 + 2.0 * x0 * x5;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -x3 * x5;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_WEDGE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 6U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = 1.0 - phi;
    const fp_t x1 = -eta - xi + 1.0;
    sfv(0) = x0 * x1;
    sfv(1) = x0 * xi;
    sfv(2) = eta * x0;
    sfv(3) = phi * x1;
    sfv(4) = phi * xi;
    sfv(5) = eta * phi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = phi - 1.0;
    const fp_t x1 = -x0;
    const fp_t x2 = -phi;
    const fp_t x3 = eta + xi - 1.0;
    sfdv(0, 0) = x0;
    sfdv(1, 0) = x0;
    sfdv(2, 0) = x3;
    sfdv(0, 1) = x1;
    sfdv(1, 1) = 0.0;
    sfdv(2, 1) = -xi;
    sfdv(0, 2) = 0.0;
    sfdv(1, 2) = x1;
    sfdv(2, 2) = -eta;
    sfdv(0, 3) = x2;
    sfdv(1, 3) = x2;
    sfdv(2, 3) = -x3;
    sfdv(0, 4) = phi;
    sfdv(1, 4) = 0.0;
    sfdv(2, 4) = xi;
    sfdv(0, 5) = 0.0;
    sfdv(1, 5) = phi;
    sfdv(2, 5) = eta;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      sfddv(0, 2) = 1;
      sfddv(1, 2) = 1;
      sfddv(2, 0) = 1;
      sfddv(2, 1) = 1;
      break;
    }
    case 1: {
      sfddv(0, 2) = -1;
      sfddv(2, 0) = -1;
      break;
    }
    case 2: {
      sfddv(1, 2) = -1;
      sfddv(2, 1) = -1;
      break;
    }
    case 3: {
      sfddv(0, 2) = -1;
      sfddv(1, 2) = -1;
      sfddv(2, 0) = -1;
      sfddv(2, 1) = -1;
      break;
    }
    case 4: {
      sfddv(0, 2) = 1;
      sfddv(2, 0) = 1;
      break;
    }
    case 5: {
      sfddv(1, 2) = 1;
      sfddv(2, 1) = 1;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_WEDGE> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 15U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    const fp_t x0 = -eta - xi + 1;
    const fp_t x1 = 2 * phi;
    const fp_t x2 = 1 - pow(x1 - 1, 2);
    const fp_t x3 = x0 * x2;
    const fp_t x4 = 0.5 * x3;
    const fp_t x5 = 2 - x1;
    const fp_t x6 = 2 * eta;
    const fp_t x7 = 2 * xi;
    const fp_t x8 = x7 - 1;
    const fp_t x9 = -x6 - x8;
    const fp_t x10 = x2 * xi;
    const fp_t x11 = 0.5 * x10;
    const fp_t x12 = x6 - 1;
    const fp_t x13 = eta * x2;
    const fp_t x14 = 0.5 * x13;
    const fp_t x15 = 1.0 * phi;
    const fp_t x16 = -x6 - x7 + 2;
    const fp_t x17 = x5 * xi;
    const fp_t x18 = 4 * eta * phi;
    sfv(0) = 0.5 * x0 * x5 * x9 - x4;
    sfv(1) = -x11 + 0.5 * x5 * x8 * xi;
    sfv(2) = 0.5 * eta * x12 * x5 - x14;
    sfv(3) = x0 * x15 * x9 - x4;
    sfv(4) = -x11 + x15 * x8 * xi;
    sfv(5) = eta * x12 * x15 - x14;
    sfv(6) = x16 * x17;
    sfv(7) = x17 * x6;
    sfv(8) = x0 * x5 * x6;
    sfv(9) = x1 * x16 * xi;
    sfv(10) = x18 * xi;
    sfv(11) = x0 * x18;
    sfv(12) = x3;
    sfv(13) = x10;
    sfv(14) = x13;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    const fp_t x0 = 2 * eta;
    const fp_t x1 = 2 * xi;
    const fp_t x2 = x1 - 1;
    const fp_t x3 = -x0 - x2;
    const fp_t x4 = 2 * phi;
    const fp_t x5 = 2 - x4;
    const fp_t x6 = 0.5 * x5;
    const fp_t x7 = -eta - xi + 1;
    const fp_t x8 = 1.0 * x5;
    const fp_t x9 = pow(x4 - 1, 2);
    const fp_t x10 = 0.5 * x9 - 0.5;
    const fp_t x11 = -x10 - x3 * x6 - x7 * x8;
    const fp_t x12 = 1.0 * phi;
    const fp_t x13 = 2.0 * phi;
    const fp_t x14 = -x10 - x12 * x3 - x13 * x7;
    const fp_t x15 = x1 * x5;
    const fp_t x16 = -x0 - x1 + 2;
    const fp_t x17 = x0 * x5;
    const fp_t x18 = 4 * phi;
    const fp_t x19 = x18 * xi;
    const fp_t x20 = eta * x18;
    const fp_t x21 = x9 - 1;
    const fp_t x22 = -x21;
    const fp_t x23 = x0 - 1;
    const fp_t x24 = 4 - 8 * phi;
    const fp_t x25 = x24 * x7;
    const fp_t x26 = 0.5 * x25;
    const fp_t x27 = 1.0 * x3 * x7;
    const fp_t x28 = x24 * xi;
    const fp_t x29 = 0.5 * x28;
    const fp_t x30 = 1.0 * x2 * xi;
    const fp_t x31 = eta * x24;
    const fp_t x32 = 0.5 * x31;
    const fp_t x33 = 1.0 * eta * x23;
    const fp_t x34 = x1 * x16;
    const fp_t x35 = 4 * eta;
    const fp_t x36 = x35 * xi;
    const fp_t x37 = x35 * x7;
    sfdv(0, 0) = x11;
    sfdv(1, 0) = x11;
    sfdv(2, 0) = -x26 - x27;
    sfdv(0, 1) = x10 + x2 * x6 + x8 * xi;
    sfdv(1, 1) = 0;
    sfdv(2, 1) = -x29 - x30;
    sfdv(0, 2) = 0;
    sfdv(1, 2) = eta * x8 + x10 + x23 * x6;
    sfdv(2, 2) = -x32 - x33;
    sfdv(0, 3) = x14;
    sfdv(1, 3) = x14;
    sfdv(2, 3) = -x26 + x27;
    sfdv(0, 4) = x10 + x12 * x2 + x13 * xi;
    sfdv(1, 4) = 0;
    sfdv(2, 4) = -x29 + x30;
    sfdv(0, 5) = 0;
    sfdv(1, 5) = eta * x13 + x10 + x12 * x23;
    sfdv(2, 5) = -x32 + x33;
    sfdv(0, 6) = -x15 + x16 * x5;
    sfdv(1, 6) = -x15;
    sfdv(2, 6) = -x34;
    sfdv(0, 7) = x17;
    sfdv(1, 7) = x15;
    sfdv(2, 7) = -x36;
    sfdv(0, 8) = -x17;
    sfdv(1, 8) = -x17 + 2 * x5 * x7;
    sfdv(2, 8) = -x37;
    sfdv(0, 9) = 2 * phi * x16 - x19;
    sfdv(1, 9) = -x19;
    sfdv(2, 9) = x34;
    sfdv(0, 10) = x20;
    sfdv(1, 10) = x19;
    sfdv(2, 10) = x36;
    sfdv(0, 11) = -x20;
    sfdv(1, 11) = 4 * phi * x7 - x20;
    sfdv(2, 11) = x37;
    sfdv(0, 12) = x21;
    sfdv(1, 12) = x21;
    sfdv(2, 12) = x25;
    sfdv(0, 13) = x22;
    sfdv(1, 13) = 0;
    sfdv(2, 13) = x28;
    sfdv(0, 14) = 0;
    sfdv(1, 14) = x22;
    sfdv(2, 14) = x31;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 - x0;
      const fp_t x2 = 4.0 * eta + 4.0 * xi;
      const fp_t x3 = -x0 - x2 + 5.0;
      sfddv(0, 0) = x1;
      sfddv(0, 1) = x1;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x1;
      sfddv(1, 1) = x1;
      sfddv(1, 2) = x3;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x3;
      sfddv(2, 2) = 4.0 - x2;
      break;
    }
    case 1: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 * xi;
      const fp_t x2 = x0 - x1 - 1.0;
      sfddv(0, 0) = 4.0 - x0;
      sfddv(0, 2) = x2;
      sfddv(2, 0) = x2;
      sfddv(2, 2) = x1;
      break;
    }
    case 2: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 * eta;
      const fp_t x2 = x0 - x1 - 1.0;
      sfddv(1, 1) = 4.0 - x0;
      sfddv(1, 2) = x2;
      sfddv(2, 1) = x2;
      sfddv(2, 2) = x1;
      break;
    }
    case 3: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 * eta + 4.0 * xi;
      const fp_t x2 = -x0 + x1 - 1.0;
      sfddv(0, 0) = x0;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x2;
      sfddv(1, 0) = x0;
      sfddv(1, 1) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x2;
      sfddv(2, 1) = x2;
      sfddv(2, 2) = 4.0 - x1;
      break;
    }
    case 4: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 * xi;
      const fp_t x2 = x0 + x1 - 3.0;
      sfddv(0, 0) = x0;
      sfddv(0, 2) = x2;
      sfddv(2, 0) = x2;
      sfddv(2, 2) = x1;
      break;
    }
    case 5: {
      const fp_t x0 = 4.0 * phi;
      const fp_t x1 = 4.0 * eta;
      const fp_t x2 = x0 + x1 - 3.0;
      sfddv(1, 1) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 1) = x2;
      sfddv(2, 2) = x1;
      break;
    }
    case 6: {
      const fp_t x0 = 4 * phi - 4;
      const fp_t x1 = 4 * eta + 8 * xi - 4;
      const fp_t x2 = 4 * xi;
      sfddv(0, 0) = 8 * phi - 8;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 7: {
      const fp_t x0 = 4 - 4 * phi;
      const fp_t x1 = -4 * eta;
      const fp_t x2 = -4 * xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 8: {
      const fp_t x0 = 4 * phi - 4;
      const fp_t x1 = 4 * eta;
      const fp_t x2 = 8 * eta + 4 * xi - 4;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 1) = 8 * phi - 8;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 9: {
      const fp_t x0 = -4 * phi;
      const fp_t x1 = -4 * eta - 8 * xi + 4;
      const fp_t x2 = -4 * xi;
      sfddv(0, 0) = -8 * phi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 10: {
      const fp_t x0 = 4 * phi;
      const fp_t x1 = 4 * eta;
      const fp_t x2 = 4 * xi;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 11: {
      const fp_t x0 = -4 * phi;
      const fp_t x1 = -4 * eta;
      const fp_t x2 = -8 * eta - 4 * xi + 4;
      sfddv(0, 1) = x0;
      sfddv(0, 2) = x1;
      sfddv(1, 0) = x0;
      sfddv(1, 1) = -8 * phi;
      sfddv(1, 2) = x2;
      sfddv(2, 0) = x1;
      sfddv(2, 1) = x2;
      break;
    }
    case 12: {
      const fp_t x0 = 8 * phi - 4;
      sfddv(0, 2) = x0;
      sfddv(1, 2) = x0;
      sfddv(2, 0) = x0;
      sfddv(2, 1) = x0;
      sfddv(2, 2) = 8 * eta + 8 * xi - 8;
      break;
    }
    case 13: {
      const fp_t x0 = 4 - 8 * phi;
      sfddv(0, 2) = x0;
      sfddv(2, 0) = x0;
      sfddv(2, 2) = -8 * xi;
      break;
    }
    case 14: {
      const fp_t x0 = 4 - 8 * phi;
      sfddv(1, 2) = x0;
      sfddv(2, 1) = x0;
      sfddv(2, 2) = -8 * eta;
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_PYRAMID> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 5U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    if (abs(1 - phi) < 1e-6) {
      sfv(0) = 0.0;
      sfv(1) = 0.0;
      sfv(2) = 0.0;
      sfv(3) = 0.0;
      sfv(4) = 1.0;
      return;
    }
    const fp_t x0 = phi - 1;
    const fp_t x1 = -x0 - xi;
    const fp_t x2 = 1.0 / (1.0 - phi);
    const fp_t x3 = x2 * (-eta - x0);
    const fp_t x4 = eta * x2;
    sfv(0) = x1 * x3;
    sfv(1) = x3 * xi;
    sfv(2) = x4 * xi;
    sfv(3) = x1 * x4;
    sfv(4) = phi;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    if (abs(1 - phi) < 1e-5) {
      sfdv(2, 0) = -1.0;
      sfdv(2, 0) = -1.0;
      sfdv(2, 0) = -1.0;
      sfdv(2, 1) = 1.0;
      sfdv(2, 1) = 0.0;
      sfdv(2, 1) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 3) = 0.0;
      sfdv(2, 3) = 1.0;
      sfdv(2, 3) = 0.0;
      sfdv(2, 4) = 0.0;
      sfdv(2, 4) = 0.0;
      sfdv(2, 4) = 1.0;
      return;
    }
    const fp_t x0 = phi - 1;
    const fp_t x1 = -eta - x0;
    const fp_t x2 = 1.0 - phi;
    const fp_t x3 = 1.0 / x2;
    const fp_t x4 = x1 * x3;
    const fp_t x5 = eta * x3;
    const fp_t x6 = -x0 - xi;
    const fp_t x7 = x3 * x6;
    const fp_t x8 = x3 * xi;
    const fp_t x9 = pow(x2, -2);
    sfdv(0, 0) = -x4;
    sfdv(1, 0) = -x7;
    sfdv(2, 0) = 1.0 * x1 * x6 * x9 - x4 - x7;
    sfdv(0, 1) = x4;
    sfdv(1, 1) = -x8;
    sfdv(2, 1) = 1.0 * x1 * x9 * xi - x8;
    sfdv(0, 2) = x5;
    sfdv(1, 2) = x8;
    sfdv(2, 2) = 1.0 * eta * x9 * xi;
    sfdv(0, 3) = -x5;
    sfdv(1, 3) = x7;
    sfdv(2, 3) = 1.0 * eta * x6 * x9 - x5;
    sfdv(0, 4) = 0;
    sfdv(1, 4) = 0;
    sfdv(2, 4) = 1;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 1.0 * x1;
      const fp_t x3 = phi - 1;
      const fp_t x4 = -eta - x3;
      const fp_t x5 = pow(x0, -2);
      const fp_t x6 = 1.0 * x5;
      const fp_t x7 = x2 - x4 * x6;
      const fp_t x8 = -x3 - xi;
      const fp_t x9 = x2 - x6 * x8;
      const fp_t x10 = 2.0 * x5;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x7;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x9;
      sfddv(2, 0) = x7;
      sfddv(2, 1) = x9;
      sfddv(2, 2) = 2.0 * x1 - x10 * x4 - x10 * x8 + 2.0 * x4 * x8 / pow(x0, 3);
      break;
    }
    case 1: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = -x1;
      const fp_t x3 = pow(x0, -2);
      const fp_t x4 = -eta - phi + 1;
      const fp_t x5 = -x1 + 1.0 * x3 * x4;
      const fp_t x6 = -1.0 * x3 * xi;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x5;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x5;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -2.0 * x3 * xi + 2.0 * x4 * xi / pow(x0, 3);
      break;
    }
    case 2: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 1.0 / pow(x0, 2);
      const fp_t x3 = eta * x2;
      const fp_t x4 = x2 * xi;
      sfddv(0, 1) = x1;
      sfddv(0, 2) = x3;
      sfddv(1, 0) = x1;
      sfddv(1, 2) = x4;
      sfddv(2, 0) = x3;
      sfddv(2, 1) = x4;
      sfddv(2, 2) = 2.0 * eta * xi / pow(x0, 3);
      break;
    }
    case 3: {
      const fp_t x0 = 1.0 - phi;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = -x1;
      const fp_t x3 = pow(x0, -2);
      const fp_t x4 = -1.0 * eta * x3;
      const fp_t x5 = -phi - xi + 1;
      const fp_t x6 = -x1 + 1.0 * x3 * x5;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x4;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x4;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -2.0 * eta * x3 + 2.0 * eta * x5 / pow(x0, 3);
      break;
    }
    case 4: {
      break;
    }
    }
    return;
  };
};
template <typename ExecSpace>
struct LagrangeSpaceGeo<ExecSpace, CellType::VTK_QUADRATIC_PYRAMID> {
public:
  static constexpr unsigned int numberOfShapeFunctions = 13U;
  static constexpr unsigned int dimensionality = 3U;
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsValues(const fp_t xi, const fp_t eta, const fp_t phi,
                             Kokkos::View<fp_t *, ExecSpace> sfv) {
    if (abs(1 - phi) < 1e-6) {
      sfv(0) = 0.0;
      sfv(1) = 0.0;
      sfv(2) = 0.0;
      sfv(3) = 0.0;
      sfv(4) = 1.0;
      sfv(5) = 0.0;
      sfv(6) = 0.0;
      sfv(7) = 0.0;
      sfv(8) = 0.0;
      sfv(9) = 0.0;
      sfv(10) = 0.0;
      sfv(11) = 0.0;
      sfv(12) = 0.0;
      return;
    }
    const fp_t x0 = 2 * phi;
    const fp_t x1 = x0 - 1;
    const fp_t x2 = 2 * eta;
    const fp_t x3 = 2 * xi;
    const fp_t x4 = x2 + x3;
    const fp_t x5 = x0 - 2;
    const fp_t x6 = x3 + x5;
    const fp_t x7 = -x6;
    const fp_t x8 = 1.0 / (phi - 1.0);
    const fp_t x9 = x8 * (-x2 - x5);
    const fp_t x10 = x7 * x9;
    const fp_t x11 = x9 * xi;
    const fp_t x12 = eta * x8;
    const fp_t x13 = 1.0 * xi;
    const fp_t x14 = x12 * x7;
    const fp_t x15 = 2.0 * x11;
    const fp_t x16 = x12 * xi;
    const fp_t x17 = 1.0 * x10;
    sfv(0) = 0.25 * x10 * (x1 + x4);
    sfv(1) = -0.5 * x11 * (-x2 + x3 - 1);
    sfv(2) = x12 * x13 * (-x0 - x4 + 3);
    sfv(3) = -0.5 * x14 * (x2 - x3 - 1);
    sfv(4) = phi * x1;
    sfv(5) = -x10 * x13;
    sfv(6) = -eta * x15;
    sfv(7) = 2.0 * x16 * x6;
    sfv(8) = -eta * x17;
    sfv(9) = -phi * x17;
    sfv(10) = -phi * x15;
    sfv(11) = -4.0 * phi * x16;
    sfv(12) = -2.0 * phi * x14;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerValues(const fp_t xi, const fp_t eta, const fp_t phi,
                                Kokkos::View<fp_t **, ExecSpace> sfdv) {
    if (abs(1 - phi) < 1e-5) {
      sfdv(2, 0) = 1.0;
      sfdv(2, 0) = 1.0;
      sfdv(2, 0) = 1.0;
      sfdv(2, 1) = -1.0;
      sfdv(2, 1) = 0.0;
      sfdv(2, 1) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 2) = 0.0;
      sfdv(2, 3) = 0.0;
      sfdv(2, 3) = -1.0;
      sfdv(2, 3) = 0.0;
      sfdv(2, 4) = 0.0;
      sfdv(2, 4) = 0.0;
      sfdv(2, 4) = 3.0;
      sfdv(2, 5) = 0.0;
      sfdv(2, 5) = 0.0;
      sfdv(2, 5) = 0.0;
      sfdv(2, 6) = 0.0;
      sfdv(2, 6) = 0.0;
      sfdv(2, 6) = 0.0;
      sfdv(2, 7) = 0.0;
      sfdv(2, 7) = 0.0;
      sfdv(2, 7) = 0.0;
      sfdv(2, 8) = 0.0;
      sfdv(2, 8) = 0.0;
      sfdv(2, 8) = 0.0;
      sfdv(2, 9) = -4.0;
      sfdv(2, 9) = -4.0;
      sfdv(2, 9) = -4.0;
      sfdv(2, 10) = 4.0;
      sfdv(2, 10) = 0.0;
      sfdv(2, 10) = 0.0;
      sfdv(2, 11) = 0.0;
      sfdv(2, 11) = 0.0;
      sfdv(2, 11) = 0.0;
      sfdv(2, 12) = 0.0;
      sfdv(2, 12) = 4.0;
      sfdv(2, 12) = 0.0;
      return;
    }
    const fp_t x0 = phi - 1.0;
    const fp_t x1 = 1.0 / x0;
    const fp_t x2 = 2 * eta;
    const fp_t x3 = 2 * phi;
    const fp_t x4 = x3 - 2;
    const fp_t x5 = -x2 - x4;
    const fp_t x6 = x1 * x5;
    const fp_t x7 = 2 * xi;
    const fp_t x8 = x4 + x7;
    const fp_t x9 = -x8;
    const fp_t x10 = 0.5 * x9;
    const fp_t x11 = x10 * x6;
    const fp_t x12 = x2 + x3 + x7;
    const fp_t x13 = x12 - 1;
    const fp_t x14 = 0.5 * x6;
    const fp_t x15 = x13 * x14;
    const fp_t x16 = x6 * xi;
    const fp_t x17 = 1.0 * x16;
    const fp_t x18 = -x2 + x7 - 1;
    const fp_t x19 = eta * x1;
    const fp_t x20 = 2.0 * x19;
    const fp_t x21 = x20 * xi;
    const fp_t x22 = 3 - x12;
    const fp_t x23 = x2 - x7 - 1;
    const fp_t x24 = 1.0 * x19;
    const fp_t x25 = x23 * x24;
    const fp_t x26 = x24 * x9;
    const fp_t x27 = 2.0 * x16;
    const fp_t x28 = 1.0 * x9;
    const fp_t x29 = -x28 * x6;
    const fp_t x30 = 2.0 * x6;
    const fp_t x31 = eta * x30;
    const fp_t x32 = 4.0 * x19;
    const fp_t x33 = x32 * xi;
    const fp_t x34 = phi * x30;
    const fp_t x35 = phi * x32;
    const fp_t x36 = x1 * x10;
    const fp_t x37 = x13 * x36;
    const fp_t x38 = x1 * xi;
    const fp_t x39 = 1.0 * x18 * x38;
    const fp_t x40 = 2.0 * x38;
    const fp_t x41 = x40 * x9;
    const fp_t x42 = -x27;
    const fp_t x43 = x20 * x9;
    const fp_t x44 = 2.0 * phi;
    const fp_t x45 = x44 * x9;
    const fp_t x46 = x1 * x45;
    const fp_t x47 = 4.0 * phi;
    const fp_t x48 = x38 * x47;
    const fp_t x49 = pow(x0, -2);
    const fp_t x50 = x49 * x5;
    const fp_t x51 = x50 * xi;
    const fp_t x52 = eta * x49;
    const fp_t x53 = x52 * xi;
    const fp_t x54 = x28 * x50;
    sfdv(0, 0) = x11 - x15;
    sfdv(1, 0) = x11 - x37;
    sfdv(2, 0) = 0.5 * x1 * x5 * x9 - 0.25 * x13 * x50 * x9 - x15 - x37;
    sfdv(0, 1) = -x14 * x18 - x17;
    sfdv(1, 1) = x17 + x39;
    sfdv(2, 1) = 0.5 * x18 * x51 + x39;
    sfdv(0, 2) = 1.0 * eta * x1 * x22 - x21;
    sfdv(1, 2) = 1.0 * x1 * x22 * xi - x21;
    sfdv(2, 2) = -x21 - 1.0 * x22 * x53;
    sfdv(0, 3) = x25 + x26;
    sfdv(1, 3) = -x23 * x36 - x26;
    sfdv(2, 3) = x10 * x23 * x52 + x25;
    sfdv(0, 4) = 0;
    sfdv(1, 4) = 0;
    sfdv(2, 4) = 4 * phi - 1;
    sfdv(0, 5) = x27 + x29;
    sfdv(1, 5) = x41;
    sfdv(2, 5) = x27 + x28 * x51 + x41;
    sfdv(0, 6) = -x31;
    sfdv(1, 6) = x33 + x42;
    sfdv(2, 6) = 2.0 * eta * x51 + x33;
    sfdv(0, 7) = x20 * x8 + x33;
    sfdv(1, 7) = x40 * x8;
    sfdv(2, 7) = x33 - 2.0 * x53 * x8;
    sfdv(0, 8) = x31;
    sfdv(1, 8) = x29 + x43;
    sfdv(2, 8) = eta * x54 + x31 + x43;
    sfdv(0, 9) = x34;
    sfdv(1, 9) = x46;
    sfdv(2, 9) = phi * x54 + x29 + x34 + x46;
    sfdv(0, 10) = -x34;
    sfdv(1, 10) = x48;
    sfdv(2, 10) = x42 + x44 * x51 + x48;
    sfdv(0, 11) = -x35;
    sfdv(1, 11) = -x48;
    sfdv(2, 11) = -x33 + x47 * x53;
    sfdv(0, 12) = x35;
    sfdv(1, 12) = -x46;
    sfdv(2, 12) = x35 - x43 + x45 * x52;
    return;
  };
  KOKKOS_INLINE_FUNCTION static void
  UpdateShapeFunctionsDerDerValues(const int_t i, const fp_t xi, const fp_t eta,
                                   const fp_t phi,
                                   Kokkos::View<fp_t **, ExecSpace> sfddv) {
    switch (i) {
    case 0: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 2 * eta;
      const fp_t x3 = 2 * phi;
      const fp_t x4 = x3 - 2;
      const fp_t x5 = -x2 - x4;
      const fp_t x6 = x1 * x5;
      const fp_t x7 = 2.0 * x6;
      const fp_t x8 = -x7;
      const fp_t x9 = 1.0 * x6;
      const fp_t x10 = 2 * xi;
      const fp_t x11 = -x10 - x4;
      const fp_t x12 = x10 + x2 + x3 - 1;
      const fp_t x13 = -1.0 * x1 * x12;
      const fp_t x14 = 1.0 * x1 * x11 + x13;
      const fp_t x15 = -x14 - x9;
      const fp_t x16 = 0.5 * x5;
      const fp_t x17 = pow(x0, -2);
      const fp_t x18 = x11 * x17;
      const fp_t x19 = x16 * x18;
      const fp_t x20 = 0.5 * x12 * x17 * x5 - x14 - x19 - x7;
      const fp_t x21 = 2.0 * x1;
      const fp_t x22 = x11 * x21;
      const fp_t x23 = -x22;
      const fp_t x24 = 0.5 * x11 * x12 * x17 - x13 - x19 - x22 - x9;
      const fp_t x25 = 1.0 * x5;
      const fp_t x26 = x12 * x17;
      sfddv(0, 0) = x8;
      sfddv(0, 1) = x15;
      sfddv(0, 2) = x20;
      sfddv(1, 0) = x15;
      sfddv(1, 1) = x23;
      sfddv(1, 2) = x24;
      sfddv(2, 0) = x20;
      sfddv(2, 1) = x24;
      sfddv(2, 2) = 1.0 * x11 * x26 + x12 * x21 - x18 * x25 + x23 + x25 * x26 +
                    x8 + x11 * x12 * x16 / pow(x0, 3);
      break;
    }
    case 1: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 2 * eta;
      const fp_t x3 = -2 * phi - x2 + 2;
      const fp_t x4 = x1 * x3;
      const fp_t x5 = -x2 + 2 * xi - 1;
      const fp_t x6 = 1.0 * x5;
      const fp_t x7 = x1 * xi;
      const fp_t x8 = 2.0 * x7;
      const fp_t x9 = x1 * x6 + x8;
      const fp_t x10 = 1.0 * x4 + x9;
      const fp_t x11 = pow(x0, -2);
      const fp_t x12 = x11 * x3;
      const fp_t x13 = 1.0 * x12 * xi;
      const fp_t x14 = 0.5 * x12 * x5 + x13 + x9;
      const fp_t x15 = x11 * xi;
      const fp_t x16 = -x13 - x15 * x6 - x8;
      sfddv(0, 0) = -2.0 * x4;
      sfddv(0, 1) = x10;
      sfddv(0, 2) = x14;
      sfddv(1, 0) = x10;
      sfddv(1, 1) = -4.0 * x7;
      sfddv(1, 2) = x16;
      sfddv(2, 0) = x14;
      sfddv(2, 1) = x16;
      sfddv(2, 2) = -2.0 * x15 * x5 - x3 * x6 * xi / pow(x0, 3);
      break;
    }
    case 2: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = eta * x1;
      const fp_t x3 = 2.0 * x2;
      const fp_t x4 = x1 * xi;
      const fp_t x5 = 2.0 * x4;
      const fp_t x6 = -2 * eta - 2 * phi - 2 * xi + 3;
      const fp_t x7 = 1.0 * x1 * x6 - x3 - x5;
      const fp_t x8 = pow(x0, -2);
      const fp_t x9 = -2.0 * eta * x8 * xi;
      const fp_t x10 = 1.0 * x6;
      const fp_t x11 = eta * x8;
      const fp_t x12 = -x10 * x11 - x3 - x9;
      const fp_t x13 = -x10 * x8 * xi - x5 - x9;
      sfddv(0, 0) = -4.0 * x2;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x12;
      sfddv(1, 0) = x7;
      sfddv(1, 1) = -4.0 * x4;
      sfddv(1, 2) = x13;
      sfddv(2, 0) = x12;
      sfddv(2, 1) = x13;
      sfddv(2, 2) = 2.0 * eta * x6 * xi / pow(x0, 3) + 4.0 * x11 * xi;
      break;
    }
    case 3: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = eta * x1;
      const fp_t x3 = 2 * xi;
      const fp_t x4 = -2 * phi - x3 + 2;
      const fp_t x5 = 1.0 * x1;
      const fp_t x6 = 2 * eta - x3 - 1;
      const fp_t x7 = 2.0 * x2;
      const fp_t x8 = x5 * x6 + x7;
      const fp_t x9 = x4 * x5 + x8;
      const fp_t x10 = pow(x0, -2);
      const fp_t x11 = x10 * x6;
      const fp_t x12 = 1.0 * eta;
      const fp_t x13 = x12 * x4;
      const fp_t x14 = x10 * x13;
      const fp_t x15 = -x11 * x12 - x14 - x7;
      const fp_t x16 = 0.5 * x11 * x4 + x14 + x8;
      sfddv(0, 0) = -4.0 * x2;
      sfddv(0, 1) = x9;
      sfddv(0, 2) = x15;
      sfddv(1, 0) = x9;
      sfddv(1, 1) = -2.0 * x1 * x4;
      sfddv(1, 2) = x16;
      sfddv(2, 0) = x15;
      sfddv(2, 1) = x16;
      sfddv(2, 2) = -2.0 * eta * x11 - x13 * x6 / pow(x0, 3);
      break;
    }
    case 4: {
      sfddv(2, 2) = 4;
      break;
    }
    case 5: {
      const fp_t x0 = 2 * phi - 2;
      const fp_t x1 = -2 * eta - x0;
      const fp_t x2 = phi - 1.0;
      const fp_t x3 = 1.0 / x2;
      const fp_t x4 = 4.0 * x3;
      const fp_t x5 = x4 * xi;
      const fp_t x6 = -x0 - 2 * xi;
      const fp_t x7 = 2.0 * x3 * x6 - x5;
      const fp_t x8 = 2.0 * x3;
      const fp_t x9 = pow(x2, -2);
      const fp_t x10 = x1 * x9;
      const fp_t x11 = 2.0 * xi;
      const fp_t x12 = x1 * x8 - x10 * x11 + 1.0 * x10 * x6 - x5 + x6 * x8;
      const fp_t x13 = x6 * x9;
      const fp_t x14 = -x11 * x13 - x5;
      const fp_t x15 = 4.0 * xi;
      sfddv(0, 0) = x1 * x4;
      sfddv(0, 1) = x7;
      sfddv(0, 2) = x12;
      sfddv(1, 0) = x7;
      sfddv(1, 2) = x14;
      sfddv(2, 0) = x12;
      sfddv(2, 1) = x14;
      sfddv(2, 2) =
          -x1 * x11 * x6 / pow(x2, 3) - x10 * x15 - x13 * x15 - 8.0 * x3 * xi;
      break;
    }
    case 6: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * x1;
      const fp_t x3 = eta * x2;
      const fp_t x4 = -2 * eta - 2 * phi + 2;
      const fp_t x5 = 2.0 * x4;
      const fp_t x6 = -x1 * x5 + x3;
      const fp_t x7 = pow(x0, -2);
      const fp_t x8 = x5 * x7;
      const fp_t x9 = eta * x8 + x3;
      const fp_t x10 = 8.0 * xi;
      const fp_t x11 = eta * x7;
      const fp_t x12 = 4.0 * xi;
      const fp_t x13 = -x11 * x12 + x2 * xi + x8 * xi;
      sfddv(0, 1) = x6;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x6;
      sfddv(1, 1) = x1 * x10;
      sfddv(1, 2) = x13;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x13;
      sfddv(2, 2) = -eta * x12 * x4 / pow(x0, 3) - x10 * x11;
      break;
    }
    case 7: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * xi;
      const fp_t x3 = x1 * x2;
      const fp_t x4 = 2 * phi + 2 * xi - 2;
      const fp_t x5 = 2.0 * x4;
      const fp_t x6 = x1 * x5 + x3;
      const fp_t x7 = pow(x0, -2);
      const fp_t x8 = eta * x7;
      const fp_t x9 = 4.0 * eta * x1 - x2 * x8 - x5 * x8;
      const fp_t x10 = x3 - x5 * x7 * xi;
      sfddv(0, 0) = 8.0 * eta * x1;
      sfddv(0, 1) = x6;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x6;
      sfddv(1, 2) = x10;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x10;
      sfddv(2, 2) = 4.0 * eta * x4 * xi / pow(x0, 3) - 8.0 * x8 * xi;
      break;
    }
    case 8: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * x1;
      const fp_t x3 = eta * x2;
      const fp_t x4 = 2 * phi - 2;
      const fp_t x5 = -2 * eta - x4;
      const fp_t x6 = 2.0 * x1 * x5 - x3;
      const fp_t x7 = pow(x0, -2);
      const fp_t x8 = x5 * x7;
      const fp_t x9 = 2.0 * eta;
      const fp_t x10 = -x3 - x8 * x9;
      const fp_t x11 = -x4 - 2 * xi;
      const fp_t x12 = 2.0 * x1;
      const fp_t x13 = x11 * x7;
      const fp_t x14 = x11 * x12 + 1.0 * x11 * x8 + x12 * x5 - x13 * x9 - x3;
      const fp_t x15 = 4.0 * eta;
      sfddv(0, 1) = x6;
      sfddv(0, 2) = x10;
      sfddv(1, 0) = x6;
      sfddv(1, 1) = x11 * x2;
      sfddv(1, 2) = x14;
      sfddv(2, 0) = x10;
      sfddv(2, 1) = x14;
      sfddv(2, 2) =
          -8.0 * eta * x1 - x13 * x15 - x15 * x8 - x11 * x5 * x9 / pow(x0, 3);
      break;
    }
    case 9: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * phi * x1;
      const fp_t x3 = -x2;
      const fp_t x4 = 2 * phi - 2;
      const fp_t x5 = -2 * eta - x4;
      const fp_t x6 = pow(x0, -2);
      const fp_t x7 = phi * x6;
      const fp_t x8 = 2.0 * x7;
      const fp_t x9 = 2.0 * x1 * x5 - x2 - x5 * x8;
      const fp_t x10 = -x4 - 2 * xi;
      const fp_t x11 = 2.0 * x1 * x10 - x10 * x8 - x2;
      const fp_t x12 = 4.0 * x7;
      sfddv(0, 1) = x3;
      sfddv(0, 2) = x9;
      sfddv(1, 0) = x3;
      sfddv(1, 2) = x11;
      sfddv(2, 0) = x9;
      sfddv(2, 1) = x11;
      sfddv(2, 2) = -8.0 * phi * x1 - 2.0 * phi * x10 * x5 / pow(x0, 3) +
                    4.0 * x1 * x10 + 4.0 * x1 * x5 - x10 * x12 +
                    2.0 * x10 * x5 * x6 - x12 * x5;
      break;
    }
    case 10: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * phi * x1;
      const fp_t x3 = -2 * eta - 2 * phi + 2;
      const fp_t x4 = 2.0 * x3;
      const fp_t x5 = pow(x0, -2);
      const fp_t x6 = phi * x5;
      const fp_t x7 = -x1 * x4 + x2 + x4 * x6;
      const fp_t x8 = 4.0 * xi;
      const fp_t x9 = 4.0 * x1 * xi - x6 * x8;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x7;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x9;
      sfddv(2, 0) = x7;
      sfddv(2, 1) = x9;
      sfddv(2, 2) = -phi * x3 * x8 / pow(x0, 3) + 8.0 * x1 * xi +
                    4.0 * x3 * x5 * xi - 8.0 * x6 * xi;
      break;
    }
    case 11: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 4.0 / x0;
      const fp_t x2 = -phi * x1;
      const fp_t x3 = pow(x0, -2);
      const fp_t x4 = 4.0 * phi;
      const fp_t x5 = -eta * x1 + eta * x3 * x4;
      const fp_t x6 = -x1 * xi + x3 * x4 * xi;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x5;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x6;
      sfddv(2, 0) = x5;
      sfddv(2, 1) = x6;
      sfddv(2, 2) = -8.0 * eta * phi * xi / pow(x0, 3) + 8.0 * eta * x3 * xi;
      break;
    }
    case 12: {
      const fp_t x0 = phi - 1.0;
      const fp_t x1 = 1.0 / x0;
      const fp_t x2 = 4.0 * phi * x1;
      const fp_t x3 = pow(x0, -2);
      const fp_t x4 = phi * x3;
      const fp_t x5 = 4.0 * eta;
      const fp_t x6 = 4.0 * eta * x1 - x4 * x5;
      const fp_t x7 = -2 * phi - 2 * xi + 2;
      const fp_t x8 = 2.0 * x7;
      const fp_t x9 = -x1 * x8 + x2 + x4 * x8;
      sfddv(0, 1) = x2;
      sfddv(0, 2) = x6;
      sfddv(1, 0) = x2;
      sfddv(1, 2) = x9;
      sfddv(2, 0) = x6;
      sfddv(2, 1) = x9;
      sfddv(2, 2) = 8.0 * eta * x1 + 4.0 * eta * x3 * x7 - 8.0 * eta * x4 -
                    phi * x5 * x7 / pow(x0, 3);
      break;
    }
    }
    return;
  };
};
} // namespace FiniteElements
} // namespace PACMAN
