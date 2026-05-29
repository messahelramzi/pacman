//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

/**
 * @file fortran_interface.cpp
 * @brief C++ implementation of the plain-C shim for the Fortran bindings.
 *
 * Each function catches all C++ exceptions and returns an integer error code
 * so that Fortran callers receive a clean C ABI with no thrown exceptions
 * crossing the language boundary.
 */

#include "fortran_interface.h"
#include "interface.hpp"

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>

extern "C" {

// ---------------------------------------------------------------------------
// Kokkos lifecycle
// ---------------------------------------------------------------------------

void pacman_kokkos_initialize_c(void) {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
  }
}

void pacman_kokkos_finalize_c(void) {
  if (Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
}

int pacman_best_execspace_c(void) {
#if defined(KOKKOS_ENABLE_HIP)
  return static_cast<int>(PACMAN::ExecSpaces::HIP);
#elif defined(KOKKOS_ENABLE_CUDA)
  return static_cast<int>(PACMAN::ExecSpaces::CUDA);
#elif defined(KOKKOS_ENABLE_SYCL)
  return static_cast<int>(PACMAN::ExecSpaces::SYCL);
#elif defined(KOKKOS_ENABLE_OPENMP)
  return static_cast<int>(PACMAN::ExecSpaces::OPENMP);
#elif defined(KOKKOS_ENABLE_THREADS)
  return static_cast<int>(PACMAN::ExecSpaces::THREADS);
#elif defined(KOKKOS_ENABLE_SERIAL)
  return static_cast<int>(PACMAN::ExecSpaces::SERIAL);
#else
  return -1;
#endif
}

// ---------------------------------------------------------------------------
// RBF-PUM interpolation
// ---------------------------------------------------------------------------

int pacman_rbf_interpolate_c(int spaceDimension, int execSpace, int rbfFunction,
                             const double *sourcePoints, int nSourcePoints,
                             const double *sourceValues,
                             const double *targetPoints, int nTargetPoints,
                             double *targetValues) {
  try {
    PACMAN::RbfInterpolateResult result = PACMAN::rbf_interpolate(
        static_cast<PACMAN::int_t>(spaceDimension),
        static_cast<unsigned char>(execSpace),
        static_cast<unsigned char>(rbfFunction),
        // The C++ API takes non-const pointers but does not modify the data.
        const_cast<PACMAN::coordinates_t *>(sourcePoints),
        static_cast<PACMAN::int_t>(nSourcePoints),
        const_cast<PACMAN::fp_t *>(sourceValues),
        const_cast<PACMAN::coordinates_t *>(targetPoints),
        static_cast<PACMAN::int_t>(nTargetPoints));

    for (int i = 0; i < nTargetPoints; ++i)
      targetValues[i] = result.targetValues[static_cast<std::size_t>(i)];

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "pacman_rbf_interpolate_c: " << e.what() << "\n";
    return -1;
  } catch (...) {
    std::cerr << "pacman_rbf_interpolate_c: unknown exception\n";
    return -2;
  }
}

// ---------------------------------------------------------------------------
// FE interpolation
// ---------------------------------------------------------------------------

int pacman_fe_interpolate_c(int spaceDimension, int execSpace, int method,
                            const double *sourcePoints, int nSourcePoints,
                            const double *sourceValues, const int *connVal,
                            int connValSize, const int *connOff,
                            int connOffSize, const int *cellTypes,
                            const double *targetPoints, int nTargetPoints,
                            double *targetValues, int *targetStatus) {
  try {
    for (int i = 0; i < connValSize; ++i)
      const_cast<int *>(
          connVal)[i]--; // Convert from Fortran 1-based to C++ 0-based indexing

    PACMAN::FeInterpolateResult result = PACMAN::fe_interpolate(
        static_cast<PACMAN::int_t>(spaceDimension),
        static_cast<unsigned char>(execSpace),
        static_cast<PACMAN::method_t>(method),
        const_cast<PACMAN::coordinates_t *>(sourcePoints),
        static_cast<PACMAN::int_t>(nSourcePoints),
        const_cast<PACMAN::fp_t *>(sourceValues),
        const_cast<PACMAN::int_t *>(connVal),
        static_cast<PACMAN::int_t>(connValSize),
        const_cast<PACMAN::offset_t *>(connOff),
        static_cast<PACMAN::int_t>(connOffSize),
        const_cast<PACMAN::cell_t *>(cellTypes),
        const_cast<PACMAN::coordinates_t *>(targetPoints),
        static_cast<PACMAN::int_t>(nTargetPoints));

    for (int i = 0; i < nTargetPoints; ++i) {
      targetValues[i] = result.targetValues[static_cast<std::size_t>(i)];
      targetStatus[i] = result.targetStatus[static_cast<std::size_t>(i)];
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "pacman_fe_interpolate_c: " << e.what() << "\n";
    return -1;
  } catch (...) {
    std::cerr << "pacman_fe_interpolate_c: unknown exception\n";
    return -2;
  }
}

// ---------------------------------------------------------------------------
// Cell-type helpers
// ---------------------------------------------------------------------------

int pacman_vtk_to_pacman_cell_type_c(const int *vtkTypes, int *pacmanTypes,
                                     int n) {
  try {
    PACMAN::vtk_to_pacman_cell_type(
        reinterpret_cast<const PACMAN::int_t *>(vtkTypes),
        reinterpret_cast<PACMAN::cell_t *>(pacmanTypes),
        static_cast<PACMAN::int_t>(n));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "pacman_vtk_to_pacman_cell_type_c: " << e.what() << "\n";
    return -1;
  } catch (...) {
    std::cerr << "pacman_vtk_to_pacman_cell_type_c: unknown exception\n";
    return -2;
  }
}

int pacman_vtk_cell_dim_c(int vtkCellType) {
  try {
    return static_cast<int>(
        PACMAN::vtk_cell_dim(static_cast<PACMAN::int_t>(vtkCellType)));
  } catch (const std::exception &e) {
    std::cerr << "pacman_vtk_cell_dim_c: " << e.what() << "\n";
    return -1;
  } catch (...) {
    std::cerr << "pacman_vtk_cell_dim_c: unknown exception\n";
    return -2;
  }
}

} // extern "C"
