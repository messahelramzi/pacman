//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "common/execspaces.hpp"
#include "common/types.hpp"
#include <vector>

namespace PACMAN {

// ---------------------------------------------------------------------------
// Execution-space constants — re-exported for convenience
// (same values as ExecSpaces members in common/execspaces.hpp).
// ---------------------------------------------------------------------------
using ExecSpaces = PACMAN::ExecSpaces;

// ---------------------------------------------------------------------------
// RBF function constants
// ---------------------------------------------------------------------------
/// @brief Constants representing available RBF basis functions as unsigned
/// chars, matching PyBindingsRbf::RbfFunctions in the Python bindings.
struct RbfFunctions {
  static constexpr unsigned char WENDLANDC0 = 0x10;
  static constexpr unsigned char WENDLANDC2 = 0x11;
  static constexpr unsigned char WENDLANDC4 = 0x12;
  static constexpr unsigned char WENDLANDC6 = 0x13;
  static constexpr unsigned char WENDLANDC8 = 0x14;
};

/// @brief Result of a finite-elements interpolation call.
struct FeInterpolateResult {
  std::vector<fp_t>
      targetValues; ///< Interpolated scalar value per target point.
  std::vector<int_t>
      targetStatus; ///< TransferStatus code per target point
                    ///< (same underlying int_t as TransferStatus enum).
};

/// @brief C++ interface for finite-elements interpolation.
///
/// Mirrors `pacman.fe.interpolate` from the Python bindings, accepting raw
/// pointers instead of NumPy arrays. All pointer arrays are row-major.
///
/// @param spaceDimension  Geometric dimension of points (1, 2, or 3).
/// @param execSpace       Execution-space selector (@see ExecSpaces constants).
/// @param method          Transfer method (@see TransferMethods enum cast to
/// method_t).
/// @param sourcePoints    Row-major source points `[nSourcePoints ×
/// spaceDimension]`.
/// @param nSourcePoints   Number of source points.
/// @param sourceValues    Source scalar values, length `nSourcePoints`.
/// @param connVal         Flattened CSR connectivity values; may be null for
///                        NEAREST_NEAREST.
/// @param connValSize     Number of entries in `connVal`.
/// @param connOff         CSR connectivity offsets; may be null for
/// NEAREST_NEAREST.
/// @param connOffSize     Number of entries in `connOff` (= nElems + 1).
/// @param cellTypes       PACMAN CellType per element, length `connOffSize -
/// 1`;
///                        use `vtk_to_pacman_cell_type` to convert from VTK
///                        IDs.
/// @param targetPoints    Row-major target points `[nTargetPoints ×
/// spaceDimension]`.
/// @param nTargetPoints   Number of target points.
/// @return FeInterpolateResult with `targetValues` and `targetStatus` of length
///         `nTargetPoints`.
/// @throws std::invalid_argument on inconsistent sizes.
/// @throws std::runtime_error if `spaceDimension` is not in {1, 2, 3}.
FeInterpolateResult
fe_interpolate(int_t spaceDimension, unsigned char execSpace, method_t method,
               coordinates_t *sourcePoints, int_t nSourcePoints,
               fp_t *sourceValues, int_t *connVal, int_t connValSize,
               offset_t *connOff, int_t connOffSize, cell_t *cellTypes,
               coordinates_t *targetPoints, int_t nTargetPoints);

/// @brief Convert an array of VTK cell type IDs to PACMAN CellType enum values.
///
/// Equivalent to `pacman.fe.vtk_to_pacman_cell_type` from the Python bindings.
///
/// @param vtkTypes    Input array of VTK cell type identifiers, length `n`.
/// @param pacmanTypes Output array of PACMAN CellType values, length `n`.
/// @param n           Number of cell types to convert.
/// @throws std::runtime_error if an unsupported VTK cell type is encountered.
void vtk_to_pacman_cell_type(const int_t *vtkTypes, cell_t *pacmanTypes,
                             int_t n);

/// @brief Return the topological dimension for a VTK cell type identifier.
///
/// Equivalent to `pacman.fe.vtk_cell_dim` from the Python bindings.
///
/// @param vtkCellType VTK cell type ID.
/// @return Topological dimension (0, 1, 2, or 3).
/// @throws std::runtime_error if the cell type is unsupported.
int_t vtk_cell_dim(int_t vtkCellType);

// ---------------------------------------------------------------------------
// RBF-PUM C++ interface
// ---------------------------------------------------------------------------

/// @brief Result of an RBF-PUM interpolation call.
struct RbfInterpolateResult {
  std::vector<fp_t>
      targetValues; ///< Interpolated scalar value per target point.
};

/// @brief C++ interface for RBF-PUM interpolation.
///
/// Mirrors `pacman.rbf.interpolate` from the Python bindings, accepting raw
/// pointers instead of NumPy arrays.  Points arrays are row-major
/// `[nPoints × spaceDimension]`.
///
/// @param spaceDimension  Geometric dimension (1, 2, or 3).
/// @param execSpace       Execution-space selector (@see ExecSpaces constants).
/// @param rbfFunction     RBF basis function selector (@see RbfFunctions
/// constants).
/// @param sourcePoints    Row-major source points `[nSourcePoints ×
/// spaceDimension]`.
/// @param nSourcePoints   Number of source points.
/// @param sourceValues    Source scalar values, length `nSourcePoints`.
/// @param targetPoints    Row-major target points `[nTargetPoints ×
/// spaceDimension]`.
/// @param nTargetPoints   Number of target points.
/// @return RbfInterpolateResult with `targetValues` of length `nTargetPoints`.
/// @throws std::invalid_argument on inconsistent sizes / null pointers.
/// @throws std::runtime_error if `spaceDimension` is not in {1, 2, 3}.
RbfInterpolateResult
rbf_interpolate(int_t spaceDimension, unsigned char execSpace,
                unsigned char rbfFunction, coordinates_t *sourcePoints,
                int_t nSourcePoints, fp_t *sourceValues,
                coordinates_t *targetPoints, int_t nTargetPoints);

} // namespace PACMAN
