//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

/**
 * @file test_unit_fe_tools.cpp
 * @brief Unit tests for FETools.hpp.
 *
 * For every LagrangeSpaceGeo specialization the following properties are
 * verified on a set of interior reference points:
 *
 *  1. Partition of unity   – sum of shape-function values == 1
 *  2. Kronecker-delta      – N_i(xi_j) == delta_ij  (nodal values)
 *  3. Derivative FD check  – analytic dN/dxi matches a centred finite
 *                            difference with h = 1e-6, tolerance 1e-5
 *  4. Second-derivative FD  – analytic d²N/(dxi_a dxi_b) from
 *                            UpdateShapeFunctionsDerDerValues matches a
 *                            centred FD on first derivatives, tol 1e-4
 *  5. IsBaryCoordInsideGeo – inside / outside / boundary predicates
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common/types.hpp"
#include "finite_elements/utils/FETools.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL [" << msg << "]: " #cond "\n";                        \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
  do {                                                                         \
    double _d = std::abs((double)(a) - (double)(b));                           \
    if (_d > (tol)) {                                                          \
      std::cerr << "FAIL [" << msg << "]: |" << (double)(a) << " - "           \
                << (double)(b) << "| = " << _d << " > " << (tol) << "\n";      \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using PACMAN::fp_t;
using PACMAN::int_t;

// ---------------------------------------------------------------------------
// Generic helpers that allocate Views and call LagrangeSpaceGeo
// ---------------------------------------------------------------------------

template <PACMAN::CellType CT>
double shapeFunctionSum(fp_t xi, fp_t eta, fp_t phi) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int n = LSG::numberOfShapeFunctions;
  Kokkos::View<fp_t *, ExecSpace> sfv("sfv", n);
  Kokkos::deep_copy(sfv, 0.0);
  LSG::UpdateShapeFunctionsValues(xi, eta, phi, sfv);
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += sfv(i);
  return s;
}

template <PACMAN::CellType CT>
fp_t shapeFunction(int idx, fp_t xi, fp_t eta, fp_t phi) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int n = LSG::numberOfShapeFunctions;
  Kokkos::View<fp_t *, ExecSpace> sfv("sfv", n);
  Kokkos::deep_copy(sfv, 0.0);
  LSG::UpdateShapeFunctionsValues(xi, eta, phi, sfv);
  return sfv(idx);
}

// Compute dN_i/dxi_dir via centred finite difference
template <PACMAN::CellType CT>
double shapeFunctionFD(int nodeIdx, int dir, fp_t xi, fp_t eta, fp_t phi,
                       double h = 1e-6) {
  auto eval = [&](fp_t dxi, fp_t deta, fp_t dphi) {
    return shapeFunction<CT>(nodeIdx, xi + dxi, eta + deta, phi + dphi);
  };
  if (dir == 0)
    return (eval(h, 0, 0) - eval(-h, 0, 0)) / (2.0 * h);
  if (dir == 1)
    return (eval(0, h, 0) - eval(0, -h, 0)) / (2.0 * h);
  return (eval(0, 0, h) - eval(0, 0, -h)) / (2.0 * h);
}

template <PACMAN::CellType CT>
double shapeFunctionDer(int dir, int nodeIdx, fp_t xi, fp_t eta, fp_t phi) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int n = LSG::numberOfShapeFunctions;
  constexpr int d = LSG::dimensionality;
  Kokkos::View<fp_t **, ExecSpace> sfdv("sfdv", d, n);
  Kokkos::deep_copy(sfdv, 0.0);
  LSG::UpdateShapeFunctionsDerValues(xi, eta, phi, sfdv);
  return sfdv(dir, nodeIdx);
}

// Analytic d²N_i / (dxi_da dxi_db) via UpdateShapeFunctionsDerDerValues
template <PACMAN::CellType CT>
double shapeFunctionDerDer(int nodeIdx, int da, int db, fp_t xi, fp_t eta,
                           fp_t phi) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int d = LSG::dimensionality;
  Kokkos::View<fp_t **, ExecSpace> sfddv("sfddv", d, d);
  Kokkos::deep_copy(sfddv, 0.0);
  LSG::UpdateShapeFunctionsDerDerValues(nodeIdx, xi, eta, phi, sfddv);
  return sfddv(da, db);
}

// FD estimate of d²N_i/(dxi_da dxi_db):
//   = (dN_i/dxi_db(xi + h*e_da) - dN_i/dxi_db(xi - h*e_da)) / (2h)
template <PACMAN::CellType CT>
double shapeFunctionDerDerFD(int nodeIdx, int da, int db, fp_t xi, fp_t eta,
                             fp_t phi, double h = 1e-5) {
  auto derAt = [&](fp_t dxi, fp_t deta, fp_t dphi) {
    return shapeFunctionDer<CT>(db, nodeIdx, xi + dxi, eta + deta, phi + dphi);
  };
  if (da == 0)
    return (derAt(h, 0, 0) - derAt(-h, 0, 0)) / (2.0 * h);
  if (da == 1)
    return (derAt(0, h, 0) - derAt(0, -h, 0)) / (2.0 * h);
  return (derAt(0, 0, h) - derAt(0, 0, -h)) / (2.0 * h);
}

// ---------------------------------------------------------------------------
// Partition-of-unity + derivative FD check for a set of interior points
// ---------------------------------------------------------------------------

template <PACMAN::CellType CT>
void checkPUAndDeriv(const std::string &name,
                     const std::vector<std::array<fp_t, 3>> &pts) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int nSF = LSG::numberOfShapeFunctions;
  constexpr int dim = LSG::dimensionality;
  constexpr double tol_pu = 1e-12;
  constexpr double tol_fd = 1e-5;

  for (auto &p : pts) {
    fp_t xi = p[0], eta = p[1], phi = p[2];

    // 1. Partition of unity
    double s = shapeFunctionSum<CT>(xi, eta, phi);
    CHECK_NEAR(s, 1.0, tol_pu, name + " PU");

    // 2. Derivative FD check
    for (int i = 0; i < nSF; ++i) {
      for (int d = 0; d < dim; ++d) {
        double analytic = shapeFunctionDer<CT>(d, i, xi, eta, phi);
        double fd = shapeFunctionFD<CT>(i, d, xi, eta, phi);
        CHECK_NEAR(analytic, fd, tol_fd,
                   name + " dN[" + std::to_string(i) + "]/dxi[" +
                       std::to_string(d) + "] at (" + std::to_string(xi) + "," +
                       std::to_string(eta) + "," + std::to_string(phi) + ")");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Kronecker-delta check: N_i(x_j) = delta_ij
// ---------------------------------------------------------------------------

template <PACMAN::CellType CT>
void checkKronecker(
    const std::string &name,
    const std::vector<std::array<fp_t, 3>> &nodes // local coords of each node
) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int nSF = LSG::numberOfShapeFunctions;
  for (int j = 0; j < nSF; ++j) {
    for (int i = 0; i < nSF; ++i) {
      fp_t val = shapeFunction<CT>(i, nodes[j][0], nodes[j][1], nodes[j][2]);
      double expected = (i == j) ? 1.0 : 0.0;
      CHECK_NEAR(val, expected, 1e-12,
                 name + " Kronecker N[" + std::to_string(i) + "](node " +
                     std::to_string(j) + ")");
    }
  }
}

// ---------------------------------------------------------------------------
// Second-derivative FD check against UpdateShapeFunctionsDerDerValues
// ---------------------------------------------------------------------------

template <PACMAN::CellType CT>
void checkDerDer(const std::string &name,
                 const std::vector<std::array<fp_t, 3>> &pts) {
  using LSG = PACMAN::FiniteElements::LagrangeSpaceGeo<ExecSpace, CT>;
  constexpr int nSF = LSG::numberOfShapeFunctions;
  constexpr int dim = LSG::dimensionality;
  // Centred FD on first derivatives has O(h²) error; h=1e-5 → error ~1e-9
  // Use a generous tolerance to accommodate floating-point cancellation.
  constexpr double tol = 1e-4;

  for (auto &p : pts) {
    fp_t xi = p[0], eta = p[1], phi = p[2];
    for (int i = 0; i < nSF; ++i) {
      for (int da = 0; da < dim; ++da) {
        for (int db = 0; db < dim; ++db) {
          double analytic = shapeFunctionDerDer<CT>(i, da, db, xi, eta, phi);
          double fd = shapeFunctionDerDerFD<CT>(i, da, db, xi, eta, phi);
          CHECK_NEAR(analytic, fd, tol,
                     name + " d2N[" + std::to_string(i) + "]/(dxi[" +
                         std::to_string(da) + "]dxi[" + std::to_string(db) +
                         "]) at (" + std::to_string(xi) + "," +
                         std::to_string(eta) + "," + std::to_string(phi) + ")");
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// IsBaryCoordInsideGeo tests
// ---------------------------------------------------------------------------

void testIsInside() {
  using namespace PACMAN::FiniteElements;
  using PACMAN::GeoSupport;

  const std::string name = "IsBaryCoordInsideGeo";

  // POINT – always inside
  CHECK(IsBaryCoordInsideGeo<GeoSupport::POINT>(0, 0, 0), name + " POINT in");

  // LINE [0,1]
  CHECK(IsBaryCoordInsideGeo<GeoSupport::LINE>(0.5, 0, 0), name + " LINE mid");
  CHECK(IsBaryCoordInsideGeo<GeoSupport::LINE>(0.0, 0, 0), name + " LINE lo");
  CHECK(IsBaryCoordInsideGeo<GeoSupport::LINE>(1.0, 0, 0), name + " LINE hi");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::LINE>(-0.1, 0, 0),
        name + " LINE out-");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::LINE>(1.1, 0, 0),
        name + " LINE out+");

  // TRIANGLE
  CHECK(IsBaryCoordInsideGeo<GeoSupport::TRIANGLE>(0.25, 0.25, 0),
        name + " TRI in");
  CHECK(IsBaryCoordInsideGeo<GeoSupport::TRIANGLE>(0.0, 0.0, 0),
        name + " TRI origin");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::TRIANGLE>(-0.1, 0.0, 0),
        name + " TRI out xi");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::TRIANGLE>(0.6, 0.6, 0),
        name + " TRI out sum");

  // QUAD [0,1]^2
  CHECK(IsBaryCoordInsideGeo<GeoSupport::QUAD>(0.5, 0.5, 0), name + " QUAD in");
  CHECK(IsBaryCoordInsideGeo<GeoSupport::QUAD>(0.0, 0.0, 0),
        name + " QUAD corner");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::QUAD>(-0.1, 0.5, 0),
        name + " QUAD out xi");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::QUAD>(0.5, 1.1, 0),
        name + " QUAD out eta");

  // TETRA
  CHECK(IsBaryCoordInsideGeo<GeoSupport::TETRA>(0.1, 0.1, 0.1),
        name + " TETRA in");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::TETRA>(0.5, 0.5, 0.5),
        name + " TETRA out sum");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::TETRA>(-0.1, 0.1, 0.1),
        name + " TETRA out xi");

  // HEXAHEDRON [0,1]^3
  CHECK(IsBaryCoordInsideGeo<GeoSupport::HEXAHEDRON>(0.5, 0.5, 0.5),
        name + " HEX in");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::HEXAHEDRON>(1.1, 0.5, 0.5),
        name + " HEX out xi");

  // WEDGE: tri in xi/eta, segment in phi
  CHECK(IsBaryCoordInsideGeo<GeoSupport::WEDGE>(0.2, 0.2, 0.5),
        name + " WEDGE in");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::WEDGE>(0.6, 0.6, 0.5),
        name + " WEDGE out tri");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::WEDGE>(0.2, 0.2, 1.1),
        name + " WEDGE out phi");

  // PYRAMID – centered: base is a diamond in (xi,eta), phi in [0,1]
  CHECK(IsBaryCoordInsideGeo<GeoSupport::PYRAMID>(0.0, 0.0, 0.3),
        name + " PYR in");
  CHECK(!IsBaryCoordInsideGeo<GeoSupport::PYRAMID>(0.5, 0.5, 1.1),
        name + " PYR out phi");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // ── IsBaryCoordInsideGeo ────────────────────────────────────────────────
    testIsInside();

    // ── VTK_LINE ────────────────────────────────────────────────────────────
    // Nodes: xi=0, xi=1
    checkKronecker<PACMAN::CellType::VTK_LINE>("VTK_LINE",
                                               {{{0.0, 0, 0}, {1.0, 0, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_LINE>(
        "VTK_LINE", {{{0.25, 0, 0}, {0.5, 0, 0}, {0.75, 0, 0}}});
    checkDerDer<PACMAN::CellType::VTK_LINE>(
        "VTK_LINE", {{{0.25, 0, 0}, {0.5, 0, 0}, {0.75, 0, 0}}});

    // ── VTK_QUADRATIC_EDGE ──────────────────────────────────────────────────
    // Nodes: xi=0, xi=1, xi=0.5
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_EDGE>(
        "VTK_QUADRATIC_EDGE", {{{0.0, 0, 0}, {1.0, 0, 0}, {0.5, 0, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_EDGE>(
        "VTK_QUADRATIC_EDGE", {{{0.25, 0, 0}, {0.5, 0, 0}, {0.75, 0, 0}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_EDGE>(
        "VTK_QUADRATIC_EDGE", {{{0.25, 0, 0}, {0.5, 0, 0}, {0.75, 0, 0}}});

    // ── VTK_TRIANGLE ────────────────────────────────────────────────────────
    // Nodes: (0,0),(1,0),(0,1)
    checkKronecker<PACMAN::CellType::VTK_TRIANGLE>(
        "VTK_TRIANGLE", {{{0.0, 0.0, 0}, {1.0, 0.0, 0}, {0.0, 1.0, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_TRIANGLE>(
        "VTK_TRIANGLE", {{{0.25, 0.25, 0}, {0.5, 0.25, 0}, {0.1, 0.1, 0}}});
    checkDerDer<PACMAN::CellType::VTK_TRIANGLE>(
        "VTK_TRIANGLE", {{{0.25, 0.25, 0}, {0.5, 0.25, 0}, {0.1, 0.1, 0}}});

    // ── VTK_QUADRATIC_TRIANGLE ──────────────────────────────────────────────
    // Nodes: (0,0),(1,0),(0,1),(0.5,0),(0.5,0.5),(0,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_TRIANGLE>(
        "VTK_QUADRATIC_TRIANGLE", {{{0.0, 0.0, 0},
                                    {1.0, 0.0, 0},
                                    {0.0, 1.0, 0},
                                    {0.5, 0.0, 0},
                                    {0.5, 0.5, 0},
                                    {0.0, 0.5, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_TRIANGLE>(
        "VTK_QUADRATIC_TRIANGLE", {{{0.25, 0.25, 0}, {0.1, 0.3, 0}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_TRIANGLE>(
        "VTK_QUADRATIC_TRIANGLE", {{{0.25, 0.25, 0}, {0.1, 0.3, 0}}});

    // ── VTK_QUAD ────────────────────────────────────────────────────────────
    // Nodes: (0,0),(1,0),(1,1),(0,1)
    checkKronecker<PACMAN::CellType::VTK_QUAD>(
        "VTK_QUAD",
        {{{0.0, 0.0, 0}, {1.0, 0.0, 0}, {1.0, 1.0, 0}, {0.0, 1.0, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUAD>(
        "VTK_QUAD", {{{0.25, 0.25, 0}, {0.5, 0.5, 0}, {0.75, 0.3, 0}}});
    checkDerDer<PACMAN::CellType::VTK_QUAD>(
        "VTK_QUAD", {{{0.25, 0.25, 0}, {0.5, 0.5, 0}, {0.75, 0.3, 0}}});

    // ── VTK_QUADRATIC_QUAD ──────────────────────────────────────────────────
    // Nodes (VTK order): corners (0,0),(1,0),(1,1),(0,1)
    //                    mid-edge (0.5,0),(1,0.5),(0.5,1),(0,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_QUAD>("VTK_QUADRATIC_QUAD",
                                                         {{{0.0, 0.0, 0},
                                                           {1.0, 0.0, 0},
                                                           {1.0, 1.0, 0},
                                                           {0.0, 1.0, 0},
                                                           {0.5, 0.0, 0},
                                                           {1.0, 0.5, 0},
                                                           {0.5, 1.0, 0},
                                                           {0.0, 0.5, 0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_QUAD>(
        "VTK_QUADRATIC_QUAD", {{{0.25, 0.25, 0}, {0.5, 0.5, 0}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_QUAD>(
        "VTK_QUADRATIC_QUAD", {{{0.25, 0.25, 0}, {0.5, 0.5, 0}}});

    // ── VTK_TETRA ───────────────────────────────────────────────────────────
    // Nodes: (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    checkKronecker<PACMAN::CellType::VTK_TETRA>(
        "VTK_TETRA",
        {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_TETRA>(
        "VTK_TETRA", {{{0.2, 0.2, 0.2}, {0.1, 0.3, 0.1}}});
    checkDerDer<PACMAN::CellType::VTK_TETRA>(
        "VTK_TETRA", {{{0.2, 0.2, 0.2}, {0.1, 0.3, 0.1}}});

    // ── VTK_HEXAHEDRON ──────────────────────────────────────────────────────
    // Nodes: all 8 corners of [0,1]^3 in VTK order
    checkKronecker<PACMAN::CellType::VTK_HEXAHEDRON>("VTK_HEXAHEDRON",
                                                     {{{0.0, 0.0, 0.0},
                                                       {1.0, 0.0, 0.0},
                                                       {1.0, 1.0, 0.0},
                                                       {0.0, 1.0, 0.0},
                                                       {0.0, 0.0, 1.0},
                                                       {1.0, 0.0, 1.0},
                                                       {1.0, 1.0, 1.0},
                                                       {0.0, 1.0, 1.0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_HEXAHEDRON>(
        "VTK_HEXAHEDRON",
        {{{0.25, 0.25, 0.25}, {0.5, 0.5, 0.5}, {0.1, 0.8, 0.4}}});
    checkDerDer<PACMAN::CellType::VTK_HEXAHEDRON>(
        "VTK_HEXAHEDRON",
        {{{0.25, 0.25, 0.25}, {0.5, 0.5, 0.5}, {0.1, 0.8, 0.4}}});

    // ── VTK_WEDGE ───────────────────────────────────────────────────────────
    // 6 nodes: tri base at phi=0 then phi=1
    // (0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1)
    checkKronecker<PACMAN::CellType::VTK_WEDGE>("VTK_WEDGE",
                                                {{{0.0, 0.0, 0.0},
                                                  {1.0, 0.0, 0.0},
                                                  {0.0, 1.0, 0.0},
                                                  {0.0, 0.0, 1.0},
                                                  {1.0, 0.0, 1.0},
                                                  {0.0, 1.0, 1.0}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_WEDGE>(
        "VTK_WEDGE", {{{0.2, 0.2, 0.5}, {0.1, 0.3, 0.7}}});
    checkDerDer<PACMAN::CellType::VTK_WEDGE>(
        "VTK_WEDGE", {{{0.2, 0.2, 0.5}, {0.1, 0.3, 0.7}}});

    // ── VTK_QUADRATIC_TETRA ─────────────────────────────────────────────────
    // Nodes (FETools order, verified symbolically):
    //   N0:(0,1,0)  N1:(0,0,0)  N2:(1,0,0)  N3:(0,0,1)
    //   N4:(0,0.5,0)  N5:(0.5,0,0)  N6:(0.5,0.5,0)
    //   N7:(0,0.5,0.5)  N8:(0,0,0.5)  N9:(0.5,0,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_TETRA>("VTK_QUADRATIC_TETRA",
                                                          {{{0.0, 1.0, 0.0},
                                                            {0.0, 0.0, 0.0},
                                                            {1.0, 0.0, 0.0},
                                                            {0.0, 0.0, 1.0},
                                                            {0.0, 0.5, 0.0},
                                                            {0.5, 0.0, 0.0},
                                                            {0.5, 0.5, 0.0},
                                                            {0.0, 0.5, 0.5},
                                                            {0.0, 0.0, 0.5},
                                                            {0.5, 0.0, 0.5}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_TETRA>(
        "VTK_QUADRATIC_TETRA", {{{0.1, 0.1, 0.1}, {0.2, 0.3, 0.1}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_TETRA>(
        "VTK_QUADRATIC_TETRA", {{{0.1, 0.1, 0.1}, {0.2, 0.3, 0.1}}});

    // ── VTK_QUADRATIC_HEXAHEDRON ─────────────────────────────────────────────
    // Nodes (VTK order, verified symbolically):
    //   Corners:
    //   (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1) Base
    //   mid-edges: (0.5,0,0),(1,0.5,0),(0.5,1,0),(0,0.5,0) Top  mid-edges:
    //   (0.5,0,1),(1,0.5,1),(0.5,1,1),(0,0.5,1) Vertical mids:
    //   (0,0,0.5),(1,0,0.5),(1,1,0.5),(0,1,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_HEXAHEDRON>(
        "VTK_QUADRATIC_HEXAHEDRON",
        {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0},
          {0.5, 0.0, 0.0}, {1.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {0.0, 0.5, 0.0},
          {0.5, 0.0, 1.0}, {1.0, 0.5, 1.0}, {0.5, 1.0, 1.0}, {0.0, 0.5, 1.0},
          {0.0, 0.0, 0.5}, {1.0, 0.0, 0.5}, {1.0, 1.0, 0.5}, {0.0, 1.0, 0.5}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_HEXAHEDRON>(
        "VTK_QUADRATIC_HEXAHEDRON", {{{0.25, 0.25, 0.25}, {0.6, 0.3, 0.7}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_HEXAHEDRON>(
        "VTK_QUADRATIC_HEXAHEDRON", {{{0.25, 0.25, 0.25}, {0.6, 0.3, 0.7}}});

    // ── VTK_QUADRATIC_WEDGE ──────────────────────────────────────────────────
    // Nodes (FETools order, verified symbolically):
    //   Tri corners phi=0: (0,0,0),(1,0,0),(0,1,0)
    //   Tri corners phi=1: (0,0,1),(1,0,1),(0,1,1)
    //   Base mid-edges phi=0: (0.5,0,0),(0.5,0.5,0),(0,0.5,0)
    //   Top  mid-edges phi=1: (0.5,0,1),(0.5,0.5,1),(0,0.5,1)
    //   Vertical mids phi=0.5: (0,0,0.5),(1,0,0.5),(0,1,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_WEDGE>("VTK_QUADRATIC_WEDGE",
                                                          {{{0.0, 0.0, 0.0},
                                                            {1.0, 0.0, 0.0},
                                                            {0.0, 1.0, 0.0},
                                                            {0.0, 0.0, 1.0},
                                                            {1.0, 0.0, 1.0},
                                                            {0.0, 1.0, 1.0},
                                                            {0.5, 0.0, 0.0},
                                                            {0.5, 0.5, 0.0},
                                                            {0.0, 0.5, 0.0},
                                                            {0.5, 0.0, 1.0},
                                                            {0.5, 0.5, 1.0},
                                                            {0.0, 0.5, 1.0},
                                                            {0.0, 0.0, 0.5},
                                                            {1.0, 0.0, 0.5},
                                                            {0.0, 1.0, 0.5}}});
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_WEDGE>(
        "VTK_QUADRATIC_WEDGE", {{{0.2, 0.2, 0.5}, {0.1, 0.3, 0.7}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_WEDGE>(
        "VTK_QUADRATIC_WEDGE", {{{0.2, 0.2, 0.5}, {0.1, 0.3, 0.7}}});

    // ── VTK_PYRAMID ─────────────────────────────────────────────────────────
    // Centered parameterization: base nodes at z=0 in a diamond,
    // apex at (0,0,1). Kronecker node coords derived from shape functions:
    //   N0:(1,0,0)  N1:(0,1,0)  N2:(-1,0,0)  N3:(0,-1,0)  N4:(0,0,1)
    checkKronecker<PACMAN::CellType::VTK_PYRAMID>("VTK_PYRAMID",
                                                  {{{1.0, 0.0, 0.0},
                                                    {0.0, 1.0, 0.0},
                                                    {-1.0, 0.0, 0.0},
                                                    {0.0, -1.0, 0.0},
                                                    {0.0, 0.0, 1.0}}});
    // Interior points well inside the pyramid, away from the apex singularity
    checkPUAndDeriv<PACMAN::CellType::VTK_PYRAMID>(
        "VTK_PYRAMID", {{{0.1, 0.1, 0.2}, {0.0, 0.0, 0.3}}});
    // (0,0,1) is the apex: the guard returns all-zero Hessian; FD differences
    // cancel to zero for every pair of directions, so the check passes.
    checkDerDer<PACMAN::CellType::VTK_PYRAMID>(
        "VTK_PYRAMID", {{{0.1, 0.1, 0.2}, {0.0, 0.0, 0.3}, {0.0, 0.0, 1.0}}});

    // ── VTK_QUADRATIC_PYRAMID ────────────────────────────────────────────────
    // Nodes (FETools order, verified symbolically):
    //   Base corners:      (0,0,0),(1,0,0),(1,1,0),(0,1,0)
    //   Apex:              (0,0,1)
    //   Base mid-edges:    (0.5,0,0),(1,0.5,0),(0.5,1,0),(0,0.5,0)
    //   Lateral mid-edges: (0,0,0.5),(0.5,0,0.5),(0.5,0.5,0.5),(0,0.5,0.5)
    checkKronecker<PACMAN::CellType::VTK_QUADRATIC_PYRAMID>(
        "VTK_QUADRATIC_PYRAMID", {{{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {1.0, 1.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0},
                                   {0.5, 0.0, 0.0},
                                   {1.0, 0.5, 0.0},
                                   {0.5, 1.0, 0.0},
                                   {0.0, 0.5, 0.0},
                                   {0.0, 0.0, 0.5},
                                   {0.5, 0.0, 0.5},
                                   {0.5, 0.5, 0.5},
                                   {0.0, 0.5, 0.5}}});
    // (0,0,1) is the apex: PU holds (guard sets N4=1), xi/eta analytic
    // derivatives are 0 (unset view) matching FD=0, phi-FD matches guard
    // limits.
    checkPUAndDeriv<PACMAN::CellType::VTK_QUADRATIC_PYRAMID>(
        "VTK_QUADRATIC_PYRAMID",
        {{{0.2, 0.2, 0.2}, {0.3, 0.3, 0.1}, {0.0, 0.0, 1.0}}});
    checkDerDer<PACMAN::CellType::VTK_QUADRATIC_PYRAMID>(
        "VTK_QUADRATIC_PYRAMID", {{{0.2, 0.2, 0.2}, {0.3, 0.3, 0.1}}});
  }
  Kokkos::finalize();

  if (g_failures == 0) {
    std::cout << "test_unit_fe_tools: ALL PASSED\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "test_unit_fe_tools: " << g_failures << " FAILURE(S)\n";
    return EXIT_FAILURE;
  }
}
