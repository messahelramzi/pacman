//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

/**
 * @file test_cpp_interface.cpp
 * @brief C++ equivalent of test_pybindings.py.
 *
 * Builds the same small two-cell 2-D mesh (VTK_QUAD + VTK_TRIANGLE),
 * converts VTK cell-type IDs to PACMAN CellType values, calls
 * `PACMAN::fe_interpolate` with INTERP_CLAMP, and checks that the
 * interpolated x-coordinates match the reference values.
 *
 * Mesh layout (same as test_pybindings.py):
 *
 *   Linear mesh:                  Quadratic mesh (for reference):
 *
 *         4                              4
 *        /\                             /\
 *       /  \       Field transfer    10/  \9
 *      /    \                         /    \
 *   3 /______\ 2   ------------->  3 /__ 7__\ 2
 *    |        |                    |        |
 *    |        |                   8|        |6
 *    |________|                    |________|
 *   0          1                  0    5    1
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "interface.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int ret = EXIT_SUCCESS;
  {
    // ------------------------------------------------------------------
    // Choose the best available execution space (mirrors Python test which
    // uses pacman.execspaces.OPENMP).  Return exit code 77 (CTest SKIP)
    // if neither OPENMP nor SERIAL is compiled in.
    // ------------------------------------------------------------------
#if defined(KOKKOS_ENABLE_OPENMP)
    constexpr unsigned char execSpace = PACMAN::ExecSpaces::OPENMP;
    const char *execSpaceName = "OPENMP";
#elif defined(KOKKOS_ENABLE_SERIAL)
    constexpr unsigned char execSpace = PACMAN::ExecSpaces::SERIAL;
    const char *execSpaceName = "SERIAL";
#else
    std::cout << "[SKIP] No host execution space available.\n";
    Kokkos::finalize();
    return 77;
#endif

    // ------------------------------------------------------------------
    // Mesh definition — two cells sharing edge [2,3]
    //   Cell 0: VTK_QUAD  (9) — vertices 0,1,2,3
    //   Cell 1: VTK_TRI   (5) — vertices 3,2,4
    // ------------------------------------------------------------------

    //  Connectivity (CSR)
    std::vector<PACMAN::int_t> connVal = {0, 1, 2, 3, 3, 2, 4};
    std::vector<PACMAN::offset_t> connOff = {0, 4, 7};

    // Convert VTK cell-type IDs to PACMAN CellType enum values
    std::vector<PACMAN::int_t> vtkCellTypes = {9, 5}; // VTK_QUAD, VTK_TRI
    std::vector<PACMAN::cell_t> cellTypes(2);
    PACMAN::vtk_to_pacman_cell_type(vtkCellTypes.data(), cellTypes.data(), 2);

    // Source points (5 nodes, 2-D)
    // clang-format off
    std::vector<PACMAN::coordinates_t> sp = {
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,
        0.5,  1.5,
    };
    // clang-format on
    const PACMAN::int_t nSP = 5;

    // sp_values = x-coordinate of each source point (sp[:, 0])
    std::vector<PACMAN::fp_t> spValues = {0.0, 1.0, 1.0, 0.0, 0.5};

    // Target points (11 nodes, 2-D)
    // clang-format off
    std::vector<PACMAN::coordinates_t> tp = {
        0.0,   0.0,
        1.0,   0.0,
        1.0,   1.0,
        0.0,   1.0,
        0.5,   1.5,
        0.5,   0.0,
        1.0,   0.5,
        0.5,   1.0,
        0.0,   0.5,
        0.25,  1.25,
        0.75,  1.25,
    };
    // clang-format on
    const PACMAN::int_t nTP = 11;

    // Reference: tp_ref = x-coordinate of each target point (tp[:, 0])
    std::vector<PACMAN::fp_t> tpRef = {0.0, 1.0, 1.0, 0.0,  0.5, 0.5,
                                       1.0, 0.5, 0.0, 0.25, 0.75};

    // ------------------------------------------------------------------
    // Call the C++ interface
    // ------------------------------------------------------------------
    const PACMAN::method_t method =
        static_cast<PACMAN::method_t>(PACMAN::TransferMethods::INTERP_CLAMP);

    PACMAN::FeInterpolateResult result = PACMAN::fe_interpolate(
        /*spaceDimension=*/2, execSpace, method, sp.data(), nSP,
        spValues.data(), connVal.data(),
        static_cast<PACMAN::int_t>(connVal.size()), connOff.data(),
        static_cast<PACMAN::int_t>(connOff.size()), cellTypes.data(), tp.data(),
        nTP);

    // ------------------------------------------------------------------
    // Validate (allclose with tolerance 1e-8, mirrors np.allclose default)
    // ------------------------------------------------------------------
    constexpr double tol = 1e-8;
    bool pass = true;
    for (PACMAN::int_t i = 0; i < nTP; ++i) {
      const double diff = std::abs(result.targetValues[i] - tpRef[i]);
      if (diff > tol) {
        std::cerr << "FAIL at index " << i << ": got " << result.targetValues[i]
                  << "  expected " << tpRef[i] << "  diff " << diff << "\n";
        pass = false;
      }
    }

    if (pass) {
      std::cout << "test_cpp_interface [" << execSpaceName
                << "] INTERP_CLAMP  PASSED\n";
    } else {
      ret = EXIT_FAILURE;
    }
  }
  Kokkos::finalize();
  return ret;
}
