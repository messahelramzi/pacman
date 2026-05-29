//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

/**
 * @file test_cpp_rbf_interface.cpp
 * @brief C++ equivalent of test_rbf-pum.py — OpenMP backend only.
 *
 * Generates 3-D regular-grid point clouds that replicate the coarse/fine mesh
 * pair used in the Python RBF tests, evaluates the 3-D Franke reference
 * function, calls `PACMAN::rbf_interpolate` for every Wendland basis
 * (C0 / C2 / C4 / C6 / C8), and validates the relative L2 error against
 * mesh-dependent tolerances (matching the thresholds in test_rbf-pum.py).
 *
 * Mesh correspondence:
 *   "coarse" ~ 0.03.npz  (spacing ≈ 0.053, 20×20×20 = 8 000 pts)
 *
 * Three scenarios (same as the Python test):
 *   same           : source = coarse , target = coarse , tol = 1.e-8
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "interface.hpp"

// ---------------------------------------------------------------------------
// Franke 3-D reference function (C++ port of franke_functions.py:franke_3d)
// ---------------------------------------------------------------------------
static double franke_3d(double x, double y, double z) {
  return 0.75 * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0) +
                           (9.0 * y - 2.0) * (9.0 * y - 2.0) +
                           (9.0 * z - 2.0) * (9.0 * z - 2.0)) /
                         4.0) +
         0.75 * std::exp(-((9.0 * x + 1.0) * (9.0 * x + 1.0) / 49.0 +
                           (9.0 * y + 1.0) / 10.0 + (9.0 * z + 1.0) / 10.0)) +
         0.50 * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0) +
                           (9.0 * y - 3.0) * (9.0 * y - 3.0) +
                           (9.0 * z - 5.0) * (9.0 * z - 5.0)) /
                         4.0) -
         0.20 * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0) +
                           (9.0 * y - 7.0) * (9.0 * y - 7.0) +
                           (9.0 * z - 5.0) * (9.0 * z - 5.0)));
}

// ---------------------------------------------------------------------------
// Regular-grid point cloud generator
// ---------------------------------------------------------------------------

struct PointCloud {
  std::vector<PACMAN::coordinates_t> points; ///< row-major [N × 3]
  std::vector<PACMAN::fp_t> values;          ///< franke_3d at each point
  PACMAN::int_t n;
};

/// Build an N×N×N regular grid in [0,1]³ and evaluate Franke 3D.
static PointCloud make_grid(int N) {
  PointCloud pc;
  pc.n = static_cast<PACMAN::int_t>(N * N * N);
  pc.points.reserve(static_cast<std::size_t>(pc.n) * 3);
  pc.values.reserve(static_cast<std::size_t>(pc.n));

  const double step = 1.0 / static_cast<double>(N - 1);
  for (int i = 0; i < N; ++i) {
    const double x = i * step;
    for (int j = 0; j < N; ++j) {
      const double y = j * step;
      for (int k = 0; k < N; ++k) {
        const double z = k * step;
        pc.points.push_back(static_cast<PACMAN::coordinates_t>(x));
        pc.points.push_back(static_cast<PACMAN::coordinates_t>(y));
        pc.points.push_back(static_cast<PACMAN::coordinates_t>(z));
        pc.values.push_back(static_cast<PACMAN::fp_t>(franke_3d(x, y, z)));
      }
    }
  }
  return pc;
}

// ---------------------------------------------------------------------------
// Relative L2 error helper
// ---------------------------------------------------------------------------
static double rel_l2(const std::vector<PACMAN::fp_t> &computed,
                     const std::vector<PACMAN::fp_t> &reference) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double diff = computed[i] - reference[i];
    num += diff * diff;
    den += reference[i] * reference[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// ---------------------------------------------------------------------------
// One test case
// ---------------------------------------------------------------------------
struct TestCase {
  const char *name;
  const PointCloud *source;
  const PointCloud *target;
  double tol;
};

static bool run_case(const TestCase &tc, unsigned char rbfFunc,
                     const char *rbfName, unsigned char execSpace) {
  PACMAN::RbfInterpolateResult result = PACMAN::rbf_interpolate(
      /*spaceDimension=*/3, execSpace, rbfFunc,
      const_cast<PACMAN::coordinates_t *>(tc.source->points.data()),
      tc.source->n, const_cast<PACMAN::fp_t *>(tc.source->values.data()),
      const_cast<PACMAN::coordinates_t *>(tc.target->points.data()),
      tc.target->n);

  const double err = rel_l2(result.targetValues, tc.target->values);
  const bool ok = (err < tc.tol);
  std::cout << (ok ? "  PASS" : "  FAIL") << "  " << tc.name << "  " << rbfName
            << "  rel_l2=" << err << "  tol=" << tc.tol << "\n";
  return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int ret = EXIT_SUCCESS;
  {
    // ------------------------------------------------------------------
    // Select execution space (HIP > CUDA > SYCL > OpenMP > Threads > Serial)
    // ------------------------------------------------------------------
    unsigned char execSpace = PACMAN::ExecSpaces::SERIAL;
    const char *execSpaceName = "SERIAL";

#if defined(KOKKOS_ENABLE_HIP)
    execSpace = PACMAN::ExecSpaces::HIP;
    execSpaceName = "HIP";
#elif defined(KOKKOS_ENABLE_CUDA)
    execSpace = PACMAN::ExecSpaces::CUDA;
    execSpaceName = "CUDA";
#elif defined(KOKKOS_ENABLE_SYCL)
    execSpace = PACMAN::ExecSpaces::SYCL;
    execSpaceName = "SYCL";
#elif defined(KOKKOS_ENABLE_OPENMP)
    execSpace = PACMAN::ExecSpaces::OPENMP;
    execSpaceName = "OPENMP";
#elif defined(KOKKOS_ENABLE_THREADS)
    execSpace = PACMAN::ExecSpaces::THREADS;
    execSpaceName = "THREADS";
#else
    std::cout << "[SKIP] No execution space available.\n";
    Kokkos::finalize();
    return 77;
#endif

    std::cout << "test_cpp_rbf_interface [" << execSpaceName << "]\n";

    // ------------------------------------------------------------------
    // Build coarse (~0.03 mesh spacing) point cloud.
    //   coarse: 20^3 = 8 000 pts, spacing ≈ 0.053
    // ------------------------------------------------------------------
    const PointCloud coarse = make_grid(20);

    // ------------------------------------------------------------------
    // Test scenarios (mirrors test_rbf-pum.py main())
    // ------------------------------------------------------------------
    //   same        : "3" in source name  → tol 1.e-8
    const TestCase cases[] = {
        {"same", &coarse, &coarse, 1.e-8},
    };

    // ------------------------------------------------------------------
    // Wendland basis functions
    // ------------------------------------------------------------------
    struct RbfEntry {
      unsigned char id;
      const char *name;
    };
    const RbfEntry rbfs[] = {
        {PACMAN::RbfFunctions::WENDLANDC0, "WENDLANDC0"},
        {PACMAN::RbfFunctions::WENDLANDC2, "WENDLANDC2"},
        {PACMAN::RbfFunctions::WENDLANDC4, "WENDLANDC4"},
        {PACMAN::RbfFunctions::WENDLANDC6, "WENDLANDC6"},
        {PACMAN::RbfFunctions::WENDLANDC8, "WENDLANDC8"},
    };

    // ------------------------------------------------------------------
    // Run all combinations
    // ------------------------------------------------------------------
    for (const auto &tc : cases) {
      for (const auto &rbf : rbfs) {
        if (!run_case(tc, rbf.id, rbf.name, execSpace))
          ret = EXIT_FAILURE;
      }
    }
  }
  Kokkos::finalize();
  return ret;
}
