//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <vector>

#include "common/transfer.hxx"
#include "common/types.hpp"

/**
 * @file FESkinTools.hpp
 * @brief Utilities and lookup tables for element faces and skin extraction.
 *
 * This header exposes helper functions that map cell types to topological
 * dimensions and provide precomputed offset/entry tables for linear,
 * triangulated, and triangulated-local face representations.
 */

namespace PACMAN {
namespace FiniteElements {

/// @brief Return topological dimension of a PACMAN cell type.
/// @param[in] type PACMAN cell type.
/// @return Dimension in `{0,1,2,3}`, or `-1` for unsupported values.
KOKKOS_INLINE_FUNCTION shortint_t getDimension(const CellType &type) {
  switch (type) {
    // LCOV_EXCL_START
  case CellType::VTK_EMPTY_CELL:
  case CellType::VTK_VERTEX:
    return 0;
    // LCOV_EXCL_STOP
  case CellType::VTK_LINE:
  case CellType::VTK_QUADRATIC_EDGE:
    return 1;
  case CellType::VTK_TRIANGLE:
  case CellType::VTK_QUAD:
  case CellType::VTK_QUADRATIC_TRIANGLE:
  case CellType::VTK_QUADRATIC_QUAD:
    return 2;
  case CellType::VTK_TETRA:
  case CellType::VTK_HEXAHEDRON:
  case CellType::VTK_WEDGE:
  case CellType::VTK_PYRAMID:
  case CellType::VTK_QUADRATIC_TETRA:
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
  case CellType::VTK_QUADRATIC_WEDGE:
  case CellType::VTK_QUADRATIC_PYRAMID:
    return 3;
  }
  // LCOV_EXCL_START
  return -1;
  // LCOV_EXCL_STOP
};

/// @brief Return the number of faces for 1D skin extraction tables.
/// @param[in] type PACMAN cell type.
/// @return Number of 1D faces, or `-1` when not applicable.
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim1(const CellType &type) {

  switch (type) {
    // LCOV_EXCL_START
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return 1;
    // LCOV_EXCL_STOP
  case CellType::VTK_LINE:
    return 2;
  case CellType::VTK_QUADRATIC_EDGE:
    return 2;
    // LCOV_EXCL_START
  case CellType::VTK_TRIANGLE:
    return -1;
  case CellType::VTK_QUAD:
    return -1;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return -1;
  case CellType::VTK_QUADRATIC_QUAD:
    return -1;
  case CellType::VTK_TETRA:
    return -1;
  case CellType::VTK_HEXAHEDRON:
    return -1;
  case CellType::VTK_WEDGE:
    return -1;
  case CellType::VTK_PYRAMID:
    return -1;
  case CellType::VTK_QUADRATIC_TETRA:
    return -1;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return -1;
  case CellType::VTK_QUADRATIC_WEDGE:
    return -1;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return -1;
  default:
    return -1;
    // LCOV_EXCL_STOP
  }
};

/// @brief Return the number of faces for 2D skin extraction tables.
/// @param[in] type PACMAN cell type.
/// @return Number of 2D faces, or `-1` when not applicable.
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim2(const CellType &type) {

  switch (type) {
    // LCOV_EXCL_START
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return -1;
  case CellType::VTK_LINE:
    return 1;
  case CellType::VTK_QUADRATIC_EDGE:
    return 1;
    // LCOV_EXCL_STOP
  case CellType::VTK_TRIANGLE:
    return 3;
  case CellType::VTK_QUAD:
    return 4;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return 3;
  case CellType::VTK_QUADRATIC_QUAD:
    return 4;
    // LCOV_EXCL_START
  case CellType::VTK_TETRA:
    return -1;
  case CellType::VTK_HEXAHEDRON:
    return -1;
  case CellType::VTK_WEDGE:
    return -1;
  case CellType::VTK_PYRAMID:
    return -1;
  case CellType::VTK_QUADRATIC_TETRA:
    return -1;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return -1;
  case CellType::VTK_QUADRATIC_WEDGE:
    return -1;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return -1;
  default:
    return -1;
    // LCOV_EXCL_STOP
  }
};

/// @brief Return the number of faces for 3D skin extraction tables.
/// @param[in] type PACMAN cell type.
/// @return Number of 3D faces, or `-1` when not applicable.
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim3(const CellType &type) {
  switch (type) {
    // LCOV_EXCL_START
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return -1;
  case CellType::VTK_LINE:
    return -1;
  case CellType::VTK_QUADRATIC_EDGE:
    return -1;
  case CellType::VTK_TRIANGLE:
    return 1;
  case CellType::VTK_QUAD:
    return 1;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return 1;
  case CellType::VTK_QUADRATIC_QUAD:
    return 1;
    // LCOV_EXCL_STOP
  case CellType::VTK_TETRA:
    return 4;
  case CellType::VTK_HEXAHEDRON:
    return 6;
  case CellType::VTK_WEDGE:
    return 5;
  case CellType::VTK_PYRAMID:
    return 5;
  case CellType::VTK_QUADRATIC_TETRA:
    return 4;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return 6;
  case CellType::VTK_QUADRATIC_WEDGE:
    return 5;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return 5;
    // LCOV_EXCL_START
  default:
    return -1;
    // LCOV_EXCL_STOP
  }
};

/// @brief Dimension-dispatched helper returning face count for one cell type.
/// @tparam Dim Requested table dimension (`1`, `2`, or `3`).
/// @param[in] type PACMAN cell type.
/// @return Number of faces for the selected dimension, or `-1` for invalid
/// cases.
template <int_t Dim>
KOKKOS_INLINE_FUNCTION shortint_t getNbFace(const CellType &type) {
  switch (Dim) {
  case 1:
    return getNbFaceDim1(type);
  case 2:
    return getNbFaceDim2(type);
  case 3:
    return getNbFaceDim3(type);
    // LCOV_EXCL_START
  default:
    assert(false);
    return -1;
    // LCOV_EXCL_STOP
  }
};
static std::vector<offset_t> elemDim1LinearFacesPaddedOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    1,
    // VTK_LINE
    3,
    // VTK_QUADRATIC_EDGE
    5,
    // VTK_TRIANGLE
    5,
    // VTK_QUAD
    5,
    // VTK_QUADRATIC_TRIANGLE
    5,
    // VTK_QUADRATIC_QUAD
    5,
    // VTK_TETRA
    5,
    // VTK_HEXAHEDRON
    5,
    // VTK_WEDGE
    5,
    // VTK_PYRAMID
    5,
    // VTK_QUADRATIC_TETRA
    5,
    // VTK_QUADRATIC_HEXAHEDRON
    5,
    // VTK_QUADRATIC_WEDGE
    5,
    // VTK_QUADRATIC_PYRAMID
    5};
static std::vector<shortint_t> elemDim1LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX
    0,
    // VTK_LINE
    0, 1,
    // VTK_QUADRATIC_EDGE
    0, 1,
    // VTK_TRIANGLE

    // VTK_QUAD

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_QUAD

    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};
static std::vector<offset_t> elemDim2LinearFacesPaddedOffsets{

    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    4,
    // VTK_TRIANGLE
    10,
    // VTK_QUAD
    18,
    // VTK_QUADRATIC_TRIANGLE
    24,
    // VTK_QUADRATIC_QUAD
    32,
    // VTK_TETRA
    32,
    // VTK_HEXAHEDRON
    32,
    // VTK_WEDGE
    32,
    // VTK_PYRAMID
    32,
    // VTK_QUADRATIC_TETRA
    32,
    // VTK_QUADRATIC_HEXAHEDRON
    32,
    // VTK_QUADRATIC_WEDGE
    32,
    // VTK_QUADRATIC_PYRAMID
    32,
};
static std::vector<shortint_t> elemDim2LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE
    /*face 0*/ 0, 1,
    // VTK_QUADRATIC_EDGE
    /*face 0*/ 0, 1,
    // VTK_TRIANGLE
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 0,
    // VTK_QUAD
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 3,
    /*face 3*/ 3, 0,
    // VTK_QUADRATIC_TRIANGLE
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 0,
    // VTK_QUADRATIC_QUAD
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 3,
    /*face 3*/ 3, 0,
    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};
static std::vector<offset_t> elemDim3LinearFacesPaddedOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_TRIANGLE
    4,
    // VTK_QUAD
    8,
    // VTK_QUADRATIC_TRIANGLE
    12,
    // VTK_QUADRATIC_QUAD
    16,
    // VTK_TETRA
    32,
    // VTK_HEXAHEDRON
    56,
    // VTK_WEDGE
    76,
    // VTK_PYRAMID
    96,
    // VTK_QUADRATIC_TETRA
    112,
    // VTK_QUADRATIC_HEXAHEDRON
    136,
    // VTK_QUADRATIC_WEDGE
    156,
    // VTK_QUADRATIC_PYRAMID
    176,
};
static std::vector<shortint_t> elemDim3LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

    // VTK_TRIANGLE
    /*face 0*/ 0, 1, 2, -1,
    // VTK_QUAD
    /*face 0*/ 0, 1, 2, 3,
    // VTK_QUADRATIC_TRIANGLE
    /*face 0*/ 0, 1, 2, -1,
    // VTK_QUADRATIC_QUAD
    /*face 0*/ 0, 1, 2, 3,
    // VTK_TETRA
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 0, 1, 3, -1,
    /*face 2*/ 1, 3, 2, -1,
    /*face 3*/ 0, 3, 2, -1,
    // VTK_HEXAHEDRON
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 4, 5, 6, 7,
    /*face 2*/ 0, 3, 7, 4,
    /*face 3*/ 1, 2, 6, 5,
    /*face 4*/ 0, 1, 5, 4,
    /*face 5*/ 3, 2, 6, 7,
    // VTK_WEDGE
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 3, 4, 5, -1,
    /*face 2*/ 0, 1, 4, 3,
    /*face 3*/ 1, 2, 5, 4,
    /*face 4*/ 0, 2, 5, 3,
    // VTK_PYRAMID
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 0, 1, 4, -1,
    /*face 2*/ 0, 4, 3, -1,
    /*face 3*/ 2, 4, 3, -1,
    /*face 4*/ 1, 2, 4, -1,
    // VTK_QUADRATIC_TETRA
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 0, 1, 3, -1,
    /*face 2*/ 1, 3, 2, -1,
    /*face 3*/ 0, 3, 2, -1,
    // VTK_QUADRATIC_HEXAHEDRON
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 4, 5, 6, 7,
    /*face 2*/ 0, 3, 7, 4,
    /*face 3*/ 1, 2, 6, 5,
    /*face 4*/ 0, 1, 5, 4,
    /*face 5*/ 3, 2, 6, 7,
    // VTK_QUADRATIC_WEDGE
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 3, 4, 5, -1,
    /*face 2*/ 0, 1, 4, 3,
    /*face 3*/ 1, 2, 5, 4,
    /*face 4*/ 0, 2, 5, 3,
    // VTK_QUADRATIC_PYRAMID
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 0, 1, 4, -1,
    /*face 2*/ 0, 4, 3, -1,
    /*face 3*/ 2, 4, 3, -1,
    /*face 4*/ 1, 2, 4, -1};

/// @brief Get padded linear-face offset table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Offset table indexed by PACMAN cell type.
inline std::vector<offset_t>
getLinearFacesPaddedOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1LinearFacesPaddedOffsets;
  case 2:
    return elemDim2LinearFacesPaddedOffsets;
  case 3:
    return elemDim3LinearFacesPaddedOffsets;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<offset_t>();
    // LCOV_EXCL_STOP
  }
};
/// @brief Get padded linear-face entry table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Flattened face-node entries indexed through the corresponding
/// offset table.
inline std::vector<shortint_t>
getLinearFacesPaddedEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1LinearFacesPaddedEntries;
  case 2:
    return elemDim2LinearFacesPaddedEntries;
  case 3:
    return elemDim3LinearFacesPaddedEntries;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<shortint_t>();
    // LCOV_EXCL_STOP
  }
};
//=== triangularization
static std::vector<offset_t> elemDim1TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    1,
    // VTK_LINE
    3,
    // VTK_QUADRATIC_EDGE
    6,
    // VTK_TRIANGLE
    6,
    // VTK_QUAD
    6,
    // VTK_QUADRATIC_TRIANGLE
    6,
    // VTK_QUADRATIC_QUAD
    6,
    // VTK_TETRA
    6,
    // VTK_HEXAHEDRON
    6,
    // VTK_WEDGE
    6,
    // VTK_PYRAMID
    6,
    // VTK_QUADRATIC_TETRA
    6,
    // VTK_QUADRATIC_HEXAHEDRON
    6,
    // VTK_QUADRATIC_WEDGE
    6,
    // VTK_QUADRATIC_PYRAMID
    6,
};

static std::vector<shortint_t> elemDim1TriFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX
    /*face0*/ 0,
    // VTK_LINE
    /*face0*/ 0,
    /*face1*/ 1,
    // VTK_QUADRATIC_EDGE
    /*face0*/ 0,
    /*face1*/ 1,
    /*face2*/ 2,
    // VTK_TRIANGLE

    // VTK_QUAD

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_QUAD

    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};

static std::vector<offset_t> elemDim2TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    6,
    // VTK_TRIANGLE
    12,
    // VTK_QUAD
    20,
    // VTK_QUADRATIC_TRIANGLE
    32,
    // VTK_QUADRATIC_QUAD
    48,
    // VTK_TETRA
    48,
    // VTK_HEXAHEDRON
    48,
    // VTK_WEDGE
    48,
    // VTK_PYRAMID
    48,
    // VTK_QUADRATIC_TETRA
    48,
    // VTK_QUADRATIC_HEXAHEDRON
    48,
    // VTK_QUADRATIC_WEDGE
    48,
    // VTK_QUADRATIC_PYRAMID
    48,
};
static std::vector<shortint_t> elemDim2TriFacesEntries{

    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE
    /*face0*/ 0, 1,
    // VTK_QUADRATIC_EDGE
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    // VTK_TRIANGLE
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    /*face2*/ 2, 0,
    // VTK_QUAD
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    /*face2*/ 2, 3,
    /*face3*/ 3, 0,
    // VTK_QUADRATIC_TRIANGLE
    /*face0*/ 0, 3,
    /*face1*/ 3, 1,
    /*face2*/ 1, 4,
    /*face3*/ 4, 2,
    /*face4*/ 2, 5,
    /*face5*/ 5, 0,
    // VTK_QUADRATIC_QUAD
    /*face0*/ 0, 4,
    /*face1*/ 4, 1,
    /*face2*/ 1, 5,
    /*face3*/ 5, 2,
    /*face4*/ 2, 6,
    /*face5*/ 6, 3,
    /*face6*/ 3, 7,
    /*face7*/ 7, 0,
    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};
static std::vector<offset_t> elemDim3TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_TRIANGLE
    3,
    // VTK_QUAD
    9,
    // VTK_QUADRATIC_TRIANGLE
    21,
    // VTK_QUADRATIC_QUAD
    39,
    // VTK_TETRA
    51,
    // VTK_HEXAHEDRON
    87,
    // VTK_WEDGE
    111,
    // VTK_PYRAMID
    129,
    // VTK_QUADRATIC_TETRA
    177,
    // VTK_QUADRATIC_HEXAHEDRON
    285,
    // VTK_QUADRATIC_WEDGE
    363,
    // VTK_QUADRATIC_PYRAMID
    429,
};
static std::vector<shortint_t> elemDim3TriFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

    // VTK_TRIANGLE
    /*face0*/ 0, 1, 2,
    // VTK_QUAD
    /*face0*/ 0, 1, 2, 0, 2, 3,
    // VTK_QUADRATIC_TRIANGLE
    /*face0*/ 0, 3, 5, 1, 4, 3, 2, 5, 4, 3, 4, 5,
    // VTK_QUADRATIC_QUAD
    /*face0*/ 0, 4, 7, 3, 7, 6, 4, 6, 7, 1, 5, 4, 2, 6, 5, 4, 5, 6,
    // VTK_TETRA
    /*face0*/ 0, 1, 2,
    /*face1*/ 0, 1, 3,
    /*face2*/ 1, 3, 2,
    /*face3*/ 0, 3, 2,
    // VTK_HEXAHEDRON
    /*face0*/ 0, 1, 2, 0, 2, 3,
    /*face1*/ 4, 5, 6, 4, 6, 7,
    /*face2*/ 0, 3, 4, 3, 7, 4,
    /*face3*/ 1, 2, 5, 2, 6, 5,
    /*face4*/ 0, 1, 4, 1, 5, 4,
    /*face5*/ 3, 2, 7, 2, 6, 7,
    // VTK_WEDGE
    /*face0*/ 0, 1, 2,
    /*face1*/ 3, 4, 5,
    /*face2*/ 0, 1, 4, 0, 4, 3,
    /*face3*/ 1, 2, 4, 2, 5, 4,
    /*face4*/ 0, 2, 3, 2, 5, 3,
    // VTK_PYRAMID
    /*face0*/ 0, 1, 2, 0, 2, 3,
    /*face1*/ 0, 1, 4,
    /*face2*/ 0, 4, 3,
    /*face3*/ 2, 4, 3,
    /*face4*/ 1, 2, 4,
    // VTK_QUADRATIC_TETRA
    /*face0*/ 0, 4, 6, 2, 6, 5, 1, 5, 4, 4, 5, 6,
    /*face1*/ 0, 4, 7, 3, 7, 8, 1, 8, 4, 4, 8, 7,
    /*face2*/ 1, 8, 5, 3, 9, 8, 2, 5, 9, 5, 8, 9,
    /*face3*/ 0, 7, 6, 2, 6, 9, 3, 9, 7, 6, 7, 9,
    // VTK_QUADRATIC_HEXAHEDRON
    /*face0*/ 0, 8, 11, 11, 10, 3, 8, 10, 11, 8, 1, 9, 2, 10, 9, 8, 9, 10,
    /*face1*/ 4, 12, 15, 15, 14, 7, 12, 14, 15, 12, 5, 13, 6, 14, 13, 12, 13,
    14,
    /*face2*/ 0, 11, 16, 16, 15, 4, 11, 15, 16, 3, 19, 11, 7, 15, 19, 11, 19,
    15,
    /*face3*/ 1, 9, 17, 17, 13, 5, 9, 13, 17, 2, 18, 9, 6, 13, 18, 9, 18, 13,
    /*face4*/ 0, 8, 16, 16, 12, 4, 8, 12, 16, 1, 17, 8, 17, 13, 12, 8, 17, 12,
    /*face5*/ 3, 10, 19, 19, 14, 7, 10, 14, 19, 2, 18, 10, 6, 14, 18, 10, 18,
    14,
    // VTK_QUADRATIC_WEDGE
    /*face0*/ 0, 6, 8, 1, 7, 6, 2, 8, 7, 6, 7, 8,
    /*face1*/ 3, 9, 11, 4, 10, 9, 5, 11, 10, 9, 10, 11,
    /*face2*/ 0, 6, 12, 12, 9, 3, 6, 9, 12, 1, 13, 6, 4, 9, 13, 6, 13, 9,
    /*face3*/ 1, 7, 13, 4, 13, 10, 7, 10, 13, 2, 14, 7, 5, 10, 14, 7, 14, 10,
    /*face4*/ 0, 8, 12, 12, 11, 3, 8, 11, 12, 2, 14, 8, 5, 11, 14, 8, 14, 11,
    // VTK_QUADRATIC_PYRAMID
    /*face0*/ 0, 5, 8, 8, 7, 3, 5, 7, 8, 1, 6, 5, 6, 2, 7, 5, 6, 7,
    /*face1*/ 0, 5, 9, 9, 10, 4, 1, 10, 5, 5, 10, 9,
    /*face2*/ 0, 9, 8, 3, 8, 12, 4, 12, 9, 8, 9, 12,
    /*face3*/ 3, 7, 12, 2, 11, 7, 4, 12, 11, 7, 11, 12,
    /*face4*/ 1, 6, 10, 2, 11, 6, 4, 10, 11, 6, 11, 10};

/// @brief Get triangulated-face offset table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Offset table for triangulated face entries.
inline std::vector<offset_t> getTriFacesOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriFacesOffsets;
  case 2:
    return elemDim2TriFacesOffsets;
  case 3:
    return elemDim3TriFacesOffsets;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<offset_t>();
    // LCOV_EXCL_STOP
  }
};
/// @brief Get triangulated-face entry table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Flattened triangulated face entries.
inline std::vector<shortint_t> getTriFacesEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriFacesEntries;
  case 2:
    return elemDim2TriFacesEntries;
  case 3:
    return elemDim3TriFacesEntries;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<shortint_t>();
    // LCOV_EXCL_STOP
  }
};
static std::vector<offset_t> elemDim1TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    2,
    // VTK_LINE
    5,
    // VTK_QUADRATIC_EDGE
    9,
    // VTK_TRIANGLE
    9,
    // VTK_QUAD
    9,
    // VTK_QUADRATIC_TRIANGLE
    9,
    // VTK_QUADRATIC_QUAD
    9,
    // VTK_TETRA
    9,
    // VTK_HEXAHEDRON
    9,
    // VTK_WEDGE
    9,
    // VTK_PYRAMID
    9,
    // VTK_QUADRATIC_TETRA
    9,
    // VTK_QUADRATIC_HEXAHEDRON
    9,
    // VTK_QUADRATIC_WEDGE
    9,
    // VTK_QUADRATIC_PYRAMID
    9,
};

static std::vector<shortint_t> elemDim1TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX
    0, 1,
    // VTK_LINE
    0, 1, 2,
    // VTK_QUADRATIC_EDGE
    0, 1, 2, 3,
    // VTK_TRIANGLE

    // VTK_QUAD

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_QUAD

    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};

static std::vector<offset_t> elemDim2TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    4,
    // VTK_TRIANGLE
    8,
    // VTK_QUAD
    13,
    // VTK_QUADRATIC_TRIANGLE
    17,
    // VTK_QUADRATIC_QUAD
    22,
    // VTK_TETRA
    22,
    // VTK_HEXAHEDRON
    22,
    // VTK_WEDGE
    22,
    // VTK_PYRAMID
    22,
    // VTK_QUADRATIC_TETRA
    22,
    // VTK_QUADRATIC_HEXAHEDRON
    22,
    // VTK_QUADRATIC_WEDGE
    22,
    // VTK_QUADRATIC_PYRAMID
    22,
};
static std::vector<shortint_t> elemDim2TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE
    0, 2,
    // VTK_QUADRATIC_EDGE
    0, 4,
    // VTK_TRIANGLE
    0, 2, 4, 6,
    // VTK_QUAD
    0, 2, 4, 6, 8,
    // VTK_QUADRATIC_TRIANGLE
    0, 4, 8, 12,
    // VTK_QUADRATIC_QUAD
    0, 4, 8, 12, 16,
    // VTK_TETRA

    // VTK_HEXAHEDRON

    // VTK_WEDGE

    // VTK_PYRAMID

    // VTK_QUADRATIC_TETRA

    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_QUADRATIC_WEDGE

    // VTK_QUADRATIC_PYRAMID

};

static std::vector<offset_t> elemDim3TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_VERTEX
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_TRIANGLE
    2,
    // VTK_QUAD
    4,
    // VTK_QUADRATIC_TRIANGLE
    6,
    // VTK_QUADRATIC_QUAD
    8,
    // VTK_TETRA
    13,
    // VTK_HEXAHEDRON
    20,
    // VTK_WEDGE
    26,
    // VTK_PYRAMID
    32,
    // VTK_QUADRATIC_TETRA
    37,
    // VTK_QUADRATIC_HEXAHEDRON
    44,
    // VTK_QUADRATIC_WEDGE
    50,
    // VTK_QUADRATIC_PYRAMID
    56,
};

static std::vector<shortint_t> elemDim3TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_VERTEX

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

    // VTK_TRIANGLE
    0, 3,
    // VTK_QUAD
    0, 6,
    // VTK_QUADRATIC_TRIANGLE
    0, 12,
    // VTK_QUADRATIC_QUAD
    0, 18,
    // VTK_TETRA
    0, 3, 6, 9, 12,
    // VTK_HEXAHEDRON
    0, 6, 12, 18, 24, 30, 36,
    // VTK_WEDGE
    0, 3, 6, 12, 18, 24,
    // VTK_PYRAMID
    0, 6, 9, 12, 15, 18,
    // VTK_QUADRATIC_TETRA
    0, 12, 24, 36, 48,
    // VTK_QUADRATIC_HEXAHEDRON
    0, 18, 36, 54, 72, 90, 108,
    // VTK_QUADRATIC_WEDGE
    0, 12, 24, 42, 60, 78,
    // VTK_QUADRATIC_PYRAMID
    0, 18, 30, 42, 54, 66};

/// @brief Get local triangulated-face offset table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Offset table for local triangulated face entries.
inline std::vector<offset_t> getTriLocFacesOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriLocFacesOffsets;
  case 2:
    return elemDim2TriLocFacesOffsets;
  case 3:
    return elemDim3TriLocFacesOffsets;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<offset_t>();
    // LCOV_EXCL_STOP
  }
};
/// @brief Get local triangulated-face entry table for a given dimension.
/// @param[in] dimension Requested dimension (`1`, `2`, or `3`).
/// @return Flattened local triangulated face entries.
inline std::vector<shortint_t> getTriLocFacesEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriLocFacesEntries;
  case 2:
    return elemDim2TriLocFacesEntries;
  case 3:
    return elemDim3TriLocFacesEntries;
    // LCOV_EXCL_START
  default:
    assert(false);
    return std::vector<shortint_t>();
    // LCOV_EXCL_STOP
  }
};

}; // namespace FiniteElements
} // namespace PACMAN
