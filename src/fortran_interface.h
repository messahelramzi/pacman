//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

/**
 * @file fortran_interface.h
 * @brief Plain-C shim declarations for the Fortran ISO C bindings interface.
 *
 * The implementation (fortran_interface.cpp) wraps PACMAN::rbf_interpolate and
 * PACMAN::fe_interpolate using extern "C" linkage so that the Fortran module
 * (pacman_fortran.f90) can bind to them with ISO_C_BINDING.
 *
 * All integer arguments use plain @c int (32-bit on every supported platform)
 * to match Fortran @c INTEGER(C_INT).  Floating-point arguments use @c double
 * to match @c REAL(C_DOUBLE).  The selector arguments (execSpace, rbfFunction,
 * method) accept the integer value of the corresponding PACMAN constant
 * defined below.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Selector constants (match PACMAN::ExecSpaces / RbfFunctions /      */
/* TransferMethods — do not rely on raw values, use these names).     */
/* ------------------------------------------------------------------ */

/** Execution space selectors, matching PACMAN::ExecSpaces. */
enum PACMAN_ExecSpace {
  PACMAN_SERIAL = 0x00,
  PACMAN_OPENMP = 0x01,
  PACMAN_THREADS = 0x02,
  PACMAN_CUDA = 0x03,
  PACMAN_HIP = 0x04,
  PACMAN_SYCL = 0x05
};

/** RBF basis-function selectors, matching PACMAN::RbfFunctions. */
enum PACMAN_RbfFunction {
  PACMAN_WENDLANDC0 = 0x10,
  PACMAN_WENDLANDC2 = 0x11,
  PACMAN_WENDLANDC4 = 0x12,
  PACMAN_WENDLANDC6 = 0x13,
  PACMAN_WENDLANDC8 = 0x14
};

/** FE method selectors, matching PACMAN::TransferMethods. */
enum PACMAN_FeMethod {
  PACMAN_FE_NEAREST_NEAREST = 0xF0,
  PACMAN_FE_INTERP_CLAMP = 0xF1,
  PACMAN_FE_INTERP_NEAREST = 0xF2,
  PACMAN_FE_INTERP_ZEROFILL = 0xF3,
  PACMAN_FE_INTERP_EXTRAP = 0xF4,
  PACMAN_MLS = 0xF6
};

/* ------------------------------------------------------------------ */
/* Kokkos lifecycle                                                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialize Kokkos with default settings.
 * Must be called before any interpolation function and only once.
 */
void pacman_kokkos_initialize_c(void);

/**
 * @brief Finalize Kokkos.
 * Must be called after all interpolation calls are complete.
 */
void pacman_kokkos_finalize_c(void);

/**
 * @brief Return the best execution space available in the current build.
 *
 * Priority order: HIP > CUDA > SYCL > OpenMP > Threads > Serial.
 * Returns -1 if no execution space is available (should not happen).
 */
int pacman_best_execspace_c(void);

/* ------------------------------------------------------------------ */
/* RBF-PUM interpolation                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief C shim for PACMAN::rbf_interpolate.
 *
 * Point arrays are row-major: point @p i occupies elements
 * @c [i*spaceDimension .. (i+1)*spaceDimension-1].
 * In Fortran, declare points as @c REAL(C_DOUBLE) :: pts(spaceDimension, n)
 * (column-major storage = C row-major storage).
 *
 * @param spaceDimension  Geometric dimension (1, 2, or 3).
 * @param execSpace       Execution-space selector (PACMAN_ExecSpace).
 * @param rbfFunction     RBF basis-function selector (PACMAN_RbfFunction).
 * @param sourcePoints    Row-major source points [nSourcePoints ×
 * spaceDimension].
 * @param nSourcePoints   Number of source points.
 * @param sourceValues    Source scalar values, length @p nSourcePoints.
 * @param targetPoints    Row-major target points [nTargetPoints ×
 * spaceDimension].
 * @param nTargetPoints   Number of target points.
 * @param targetValues    Caller-allocated output, length @p nTargetPoints.
 * @return 0 on success, non-zero on error.
 */
int pacman_rbf_interpolate_c(int spaceDimension, int execSpace, int rbfFunction,
                             const double *sourcePoints, int nSourcePoints,
                             const double *sourceValues,
                             const double *targetPoints, int nTargetPoints,
                             double *targetValues);

/* ------------------------------------------------------------------ */
/* FE interpolation                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief C shim for PACMAN::fe_interpolate.
 *
 * Connectivity arrays use CSR format (int32 values).  For NEAREST_NEAREST,
 * pass NULL for @p connVal, @p connOff, @p cellTypes and zero for the
 * corresponding size arguments.
 *
 * @param spaceDimension  Geometric dimension (1, 2, or 3).
 * @param execSpace       Execution-space selector.
 * @param method          FE method selector (PACMAN_FeMethod).
 * @param sourcePoints    Row-major source points [nSourcePoints ×
 * spaceDimension].
 * @param nSourcePoints   Number of source points.
 * @param sourceValues    Source scalar values, length @p nSourcePoints.
 * @param connVal         CSR connectivity values.
 * @param connValSize     Number of entries in @p connVal.
 * @param connOff         CSR connectivity offsets.
 * @param connOffSize     Number of entries in @p connOff (= nElems + 1).
 * @param cellTypes       PACMAN CellType per element, length @p connOffSize
 * - 1.
 * @param targetPoints    Row-major target points [nTargetPoints ×
 * spaceDimension].
 * @param nTargetPoints   Number of target points.
 * @param targetValues    Caller-allocated output, length @p nTargetPoints.
 * @param targetStatus    Caller-allocated status output, length @p
 * nTargetPoints.
 * @return 0 on success, non-zero on error.
 */
int pacman_fe_interpolate_c(int spaceDimension, int execSpace, int method,
                            const double *sourcePoints, int nSourcePoints,
                            const double *sourceValues, const int *connVal,
                            int connValSize, const int *connOff,
                            int connOffSize, const int *cellTypes,
                            const double *targetPoints, int nTargetPoints,
                            double *targetValues, int *targetStatus);

/* ------------------------------------------------------------------ */
/* MLS interpolation                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief C shim for PACMAN::mls_interpolate.
 *
 * @param spaceDimension  Geometric dimension (1, 2, or 3).
 * @param execSpace       Execution-space selector (PACMAN_ExecSpace).
 * @param sourcePoints    Row-major source points [nSourcePoints ×
 * spaceDimension].
 * @param nSourcePoints   Number of source points.
 * @param sourceValues    Source scalar values, length @p nSourcePoints.
 * @param targetPoints    Row-major target points [nTargetPoints ×
 * spaceDimension].
 * @param nTargetPoints   Number of target points.
 * @param targetValues    Caller-allocated output, length @p nTargetPoints.
 * @return 0 on success, non-zero on error.
 */
int pacman_mls_interpolate_c(int spaceDimension, int execSpace,
                             const double *sourcePoints, int nSourcePoints,
                             const double *sourceValues,
                             const double *targetPoints, int nTargetPoints,
                             double *targetValues);

/* ------------------------------------------------------------------ */
/* Cell-type helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Convert an array of VTK cell-type IDs to PACMAN CellType values.
 *
 * Equivalent to PACMAN::vtk_to_pacman_cell_type.
 *
 * @param vtkTypes    Input VTK cell-type IDs, length @p n.
 * @param pacmanTypes Output PACMAN cell-type codes, length @p n.
 * @param n           Number of cell types to convert.
 * @return 0 on success, non-zero if an unsupported VTK type is encountered.
 */
int pacman_vtk_to_pacman_cell_type_c(const int *vtkTypes, int *pacmanTypes,
                                     int n);

/**
 * @brief Return the topological dimension for a VTK cell-type ID.
 *
 * Equivalent to PACMAN::vtk_cell_dim.
 *
 * @param vtkCellType VTK cell-type ID.
 * @return Topological dimension (0, 1, 2, or 3), or -1 on error.
 */
int pacman_vtk_cell_dim_c(int vtkCellType);

#ifdef __cplusplus
}
#endif
