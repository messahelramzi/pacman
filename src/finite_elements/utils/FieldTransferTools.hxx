#pragma once

#include <vector>

#include "common/transfer.hxx"
#include "common/types.hpp"

namespace PACMAN {
namespace FiniteElement {

KOKKOS_INLINE_FUNCTION shortint_t getDimension(const CellType &type) {
  switch (type) {
  case CellType::VTK_LINE:
  case CellType::VTK_QUADRATIC_EDGE:
    return 1;
  case CellType::VTK_QUAD:
  case CellType::VTK_QUADRATIC_QUAD:
  case CellType::VTK_TRIANGLE:
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return 2;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
  case CellType::VTK_HEXAHEDRON:
  case CellType::VTK_QUADRATIC_PYRAMID:
  case CellType::VTK_PYRAMID:
  case CellType::VTK_QUADRATIC_TETRA:
  case CellType::VTK_TETRA:
  case CellType::VTK_QUADRATIC_WEDGE:
  case CellType::VTK_WEDGE:
    return 3;
  }
  return -1;
};
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim1(const CellType &type) {
  switch (type) {
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return 1;
  case CellType::VTK_LINE:
    return 2;
  case CellType::VTK_QUADRATIC_EDGE:
    return 2;
  case CellType::VTK_QUAD:
    return -1;
  case CellType::VTK_QUADRATIC_QUAD:
    return -1;
  case CellType::VTK_TRIANGLE:
    return -1;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return -1;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return -1;
  case CellType::VTK_HEXAHEDRON:
    return -1;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return -1;
  case CellType::VTK_PYRAMID:
    return -1;
  case CellType::VTK_QUADRATIC_TETRA:
    return -1;
  case CellType::VTK_TETRA:
    return -1;
  case CellType::VTK_QUADRATIC_WEDGE:
    return -1;
  case CellType::VTK_WEDGE:
    return -1;
  default:
    return -1;
  }
};
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim2(const CellType &type) {
  switch (type) {
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return -1;
  case CellType::VTK_LINE:
    return 1;
  case CellType::VTK_QUADRATIC_EDGE:
    return 1;
  case CellType::VTK_QUAD:
    return 4;
  case CellType::VTK_QUADRATIC_QUAD:
    return 4;
  case CellType::VTK_TRIANGLE:
    return 3;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return 3;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return -1;
  case CellType::VTK_HEXAHEDRON:
    return -1;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return -1;
  case CellType::VTK_PYRAMID:
    return -1;
  case CellType::VTK_QUADRATIC_TETRA:
    return -1;
  case CellType::VTK_TETRA:
    return -1;
  case CellType::VTK_QUADRATIC_WEDGE:
    return -1;
  case CellType::VTK_WEDGE:
    return -1;
  default:
    return -1;
  }
};
KOKKOS_INLINE_FUNCTION shortint_t getNbFaceDim3(const CellType &type) {
  switch (type) {
  case CellType::VTK_EMPTY_CELL:
    return -1;
  case CellType::VTK_VERTEX:
    return -1;
  case CellType::VTK_LINE:
    return -1;
  case CellType::VTK_QUADRATIC_EDGE:
    return -1;
  case CellType::VTK_QUAD:
    return 1;
  case CellType::VTK_QUADRATIC_QUAD:
    return 1;
  case CellType::VTK_TRIANGLE:
    return 1;
  case CellType::VTK_QUADRATIC_TRIANGLE:
    return 1;
  case CellType::VTK_QUADRATIC_HEXAHEDRON:
    return 6;
  case CellType::VTK_HEXAHEDRON:
    return 6;
  case CellType::VTK_QUADRATIC_PYRAMID:
    return 5;
  case CellType::VTK_PYRAMID:
    return 5;
  case CellType::VTK_QUADRATIC_TETRA:
    return 4;
  case CellType::VTK_TETRA:
    return 4;
  case CellType::VTK_QUADRATIC_WEDGE:
    return 5;
  case CellType::VTK_WEDGE:
    return 5;
  default:
    return -1;
  }
};
template <int_t spaceDimension>
KOKKOS_INLINE_FUNCTION shortint_t getNbFace(const CellType &type) {
  switch (spaceDimension) {
  case 1:
    return getNbFaceDim1(type);
  case 2:
    return getNbFaceDim2(type);
  case 3:
    return getNbFaceDim3(type);
  default:
    assert(false);
    return -1;
  }
};
static std::vector<offset_t> elemDim1LinearFacesPaddedOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    4,
    // VTK_QUADRATIC_HEXAHEDRON
    2,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    2,
    // VTK_HEXAHEDRON
    2,
    // VTK_VERTEX
    5,
    // VTK_QUADRATIC_PYRAMID
    2,
    // VTK_BIQUADRATIC_PYRAMID
    2,
    // VTK_PYRAMID
    2,
    // VTK_QUAD
    2,
    // VTK_QUADRATIC_QUAD
    2,
    // VTK_BIQUADRATIC_QUAD
    2,
    // VTK_QUADRATIC_TETRA
    2,
    // VTK_TETRA
    2,
    // VTK_TRIANGLE
    2,
    // VTK_QUADRATIC_TRIANGLE
    2,
    // VTK_QUADRATIC_WEDGE
    2,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    2,
    // VTK_WEDGE
    2,
};
static std::vector<shortint_t> elemDim1LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    0, 1,
    // VTK_QUADRATIC_EDGE
    0, 1,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX
    0,
    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD

    // VTK_QUADRATIC_QUAD

    // VTK_BIQUADRATIC_QUAD

    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim2LinearFacesPaddedOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    4,
    // VTK_QUADRATIC_HEXAHEDRON
    4,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    4,
    // VTK_HEXAHEDRON
    4,
    // VTK_VERTEX
    4,
    // VTK_QUADRATIC_PYRAMID
    4,
    // VTK_BIQUADRATIC_PYRAMID
    4,
    // VTK_PYRAMID
    4,
    // VTK_QUAD
    12,
    // VTK_QUADRATIC_QUAD
    20,
    // VTK_BIQUADRATIC_QUAD
    28,
    // VTK_QUADRATIC_TETRA
    28,
    // VTK_TETRA
    28,
    // VTK_TRIANGLE
    34,
    // VTK_QUADRATIC_TRIANGLE
    40,
    // VTK_QUADRATIC_WEDGE
    40,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    40,
    // VTK_WEDGE
    40,
};
static std::vector<shortint_t> elemDim2LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    /*face 0*/ 0, 1,
    // VTK_QUADRATIC_EDGE
    /*face 0*/ 0, 1,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 3,
    /*face 3*/ 3, 0,
    // VTK_QUADRATIC_QUAD
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 3,
    /*face 3*/ 3, 0,
    // VTK_BIQUADRATIC_QUAD
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 3,
    /*face 3*/ 3, 0,
    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 0,
    // VTK_QUADRATIC_TRIANGLE
    /*face 0*/ 0, 1,
    /*face 1*/ 1, 2,
    /*face 2*/ 2, 0,
    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim3LinearFacesPaddedOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_QUADRATIC_HEXAHEDRON
    24,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    48,
    // VTK_HEXAHEDRON
    72,
    // VTK_VERTEX
    72,
    // VTK_QUADRATIC_PYRAMID
    92,
    // VTK_BIQUADRATIC_PYRAMID
    112,
    // VTK_PYRAMID
    132,
    // VTK_QUAD
    136,
    // VTK_QUADRATIC_QUAD
    140,
    // VTK_BIQUADRATIC_QUAD
    144,
    // VTK_QUADRATIC_TETRA
    160,
    // VTK_TETRA
    176,
    // VTK_TRIANGLE
    180,
    // VTK_QUADRATIC_TRIANGLE
    184,
    // VTK_QUADRATIC_WEDGE
    208,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    232,
    // VTK_WEDGE
    256,
};
static std::vector<shortint_t> elemDim3LinearFacesPaddedEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

    // VTK_QUADRATIC_HEXAHEDRON
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 4, 5, 6, 7,
    /*face 2*/ 0, 3, 7, 4,
    /*face 3*/ 1, 2, 6, 5,
    /*face 4*/ 0, 1, 5, 4,
    /*face 5*/ 3, 2, 6, 7,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 4, 5, 6, 7,
    /*face 2*/ 0, 3, 7, 4,
    /*face 3*/ 1, 2, 6, 5,
    /*face 4*/ 0, 1, 5, 4,
    /*face 5*/ 3, 2, 6, 7,
    // VTK_HEXAHEDRON
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 4, 5, 6, 7,
    /*face 2*/ 0, 3, 7, 4,
    /*face 3*/ 1, 2, 6, 5,
    /*face 4*/ 0, 1, 5, 4,
    /*face 5*/ 3, 2, 6, 7,
    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 0, 1, 4, -1,
    /*face 2*/ 0, 4, 3, -1,
    /*face 3*/ 2, 4, 3, -1,
    /*face 4*/ 1, 2, 4, -1,
    // VTK_BIQUADRATIC_PYRAMID
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 0, 1, 4, -1,
    /*face 2*/ 0, 4, 3, -1,
    /*face 3*/ 2, 4, 3, -1,
    /*face 4*/ 1, 2, 4, -1,
    // VTK_PYRAMID
    /*face 0*/ 0, 1, 2, 3,
    /*face 1*/ 0, 1, 4, -1,
    /*face 2*/ 0, 4, 3, -1,
    /*face 3*/ 2, 4, 3, -1,
    /*face 4*/ 1, 2, 4, -1,
    // VTK_QUAD
    /*face 0*/ 0, 1, 2, 3,
    // VTK_QUADRATIC_QUAD
    /*face 0*/ 0, 1, 2, 3,
    // VTK_BIQUADRATIC_QUAD
    /*face 0*/ 0, 1, 2, 3,
    // VTK_QUADRATIC_TETRA
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 0, 1, 3, -1,
    /*face 2*/ 1, 3, 2, -1,
    /*face 3*/ 0, 3, 2, -1,
    // VTK_TETRA
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 0, 1, 3, -1,
    /*face 2*/ 1, 3, 2, -1,
    /*face 3*/ 0, 3, 2, -1,
    // VTK_TRIANGLE
    /*face 0*/ 0, 1, 2, -1,
    // VTK_QUADRATIC_TRIANGLE
    /*face 0*/ 0, 1, 2, -1,
    // VTK_QUADRATIC_WEDGE
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 3, 4, 5, -1,
    /*face 2*/ 0, 1, 4, 3,
    /*face 3*/ 1, 2, 5, 4,
    /*face 4*/ 0, 2, 5, 3,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 3, 4, 5, -1,
    /*face 2*/ 0, 1, 4, 3,
    /*face 3*/ 1, 2, 5, 4,
    /*face 4*/ 0, 2, 5, 3,
    // VTK_WEDGE
    /*face 0*/ 0, 1, 2, -1,
    /*face 1*/ 3, 4, 5, -1,
    /*face 2*/ 0, 1, 4, 3,
    /*face 3*/ 1, 2, 5, 4,
    /*face 4*/ 0, 2, 5, 3};
inline std::vector<offset_t>
getLinearFacesPaddedOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1LinearFacesPaddedOffsets;
  case 2:
    return elemDim2LinearFacesPaddedOffsets;
  case 3:
    return elemDim3LinearFacesPaddedOffsets;
  default:
    assert(false);
    return std::vector<offset_t>();
  }
};
inline std::vector<shortint_t>
getLinearFacesPaddedEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1LinearFacesPaddedEntries;
  case 2:
    return elemDim2LinearFacesPaddedEntries;
  case 3:
    return elemDim3LinearFacesPaddedEntries;
  default:
    assert(false);
    return std::vector<shortint_t>();
  }
};
//=== triangularization
static std::vector<offset_t> elemDim1TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    5,
    // VTK_QUADRATIC_HEXAHEDRON
    5,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    5,
    // VTK_HEXAHEDRON
    5,
    // VTK_VERTEX
    6,
    // VTK_QUADRATIC_PYRAMID
    6,
    // VTK_BIQUADRATIC_PYRAMID
    6,
    // VTK_PYRAMID
    6,
    // VTK_QUAD
    6,
    // VTK_QUADRATIC_QUAD
    6,
    // VTK_BIQUADRATIC_QUAD
    6,
    // VTK_QUADRATIC_TETRA
    6,
    // VTK_TETRA
    6,
    // VTK_TRIANGLE
    6,
    // VTK_QUADRATIC_TRIANGLE
    6,
    // VTK_QUADRATIC_WEDGE
    6,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    6,
    // VTK_WEDGE
    6,
};
static std::vector<shortint_t> elemDim1TriFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    /*face0*/ 0,
    /*face1*/ 1,
    // VTK_QUADRATIC_EDGE
    /*face0*/ 0,
    /*face1*/ 1,
    /*face2*/ 2,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX
    /*face0*/ 0,
    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD

    // VTK_QUADRATIC_QUAD

    // VTK_BIQUADRATIC_QUAD

    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim2TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    6,
    // VTK_QUADRATIC_HEXAHEDRON
    6,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    6,
    // VTK_HEXAHEDRON
    6,
    // VTK_VERTEX
    6,
    // VTK_QUADRATIC_PYRAMID
    6,
    // VTK_BIQUADRATIC_PYRAMID
    6,
    // VTK_PYRAMID
    6,
    // VTK_QUAD
    14,
    // VTK_QUADRATIC_QUAD
    30,
    // VTK_BIQUADRATIC_QUAD
    46,
    // VTK_QUADRATIC_TETRA
    46,
    // VTK_TETRA
    46,
    // VTK_TRIANGLE
    52,
    // VTK_QUADRATIC_TRIANGLE
    64,
    // VTK_QUADRATIC_WEDGE
    64,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    64,
    // VTK_WEDGE
    64,
};
static std::vector<shortint_t> elemDim2TriFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    /*face0*/ 0, 1,
    // VTK_QUADRATIC_EDGE
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    /*face2*/ 2, 3,
    /*face3*/ 3, 0,
    // VTK_QUADRATIC_QUAD
    /*face0*/ 0, 4,
    /*face1*/ 4, 1,
    /*face2*/ 1, 5,
    /*face3*/ 5, 2,
    /*face4*/ 2, 6,
    /*face5*/ 6, 3,
    /*face6*/ 3, 7,
    /*face7*/ 7, 0,
    // VTK_BIQUADRATIC_QUAD
    /*face0*/ 0, 4,
    /*face1*/ 4, 1,
    /*face2*/ 1, 5,
    /*face3*/ 5, 2,
    /*face4*/ 2, 6,
    /*face5*/ 6, 3,
    /*face6*/ 3, 7,
    /*face7*/ 7, 0,
    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE
    /*face0*/ 0, 1,
    /*face1*/ 1, 2,
    /*face2*/ 2, 0,
    // VTK_QUADRATIC_TRIANGLE
    /*face0*/ 0, 3,
    /*face1*/ 3, 1,
    /*face2*/ 1, 4,
    /*face3*/ 4, 2,
    /*face4*/ 2, 5,
    /*face5*/ 5, 0,
    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim3TriFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_QUADRATIC_HEXAHEDRON
    108,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    216,
    // VTK_HEXAHEDRON
    252,
    // VTK_VERTEX
    252,
    // VTK_QUADRATIC_PYRAMID
    318,
    // VTK_BIQUADRATIC_PYRAMID
    384,
    // VTK_PYRAMID
    402,
    // VTK_QUAD
    408,
    // VTK_QUADRATIC_QUAD
    420,
    // VTK_BIQUADRATIC_QUAD
    438,
    // VTK_QUADRATIC_TETRA
    486,
    // VTK_TETRA
    498,
    // VTK_TRIANGLE
    501,
    // VTK_QUADRATIC_TRIANGLE
    513,
    // VTK_QUADRATIC_WEDGE
    591,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    609,
    // VTK_WEDGE
    633,
};
static std::vector<shortint_t> elemDim3TriFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

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
    // VTK_TRIQUADRATIC_HEXAHEDRON
    /*face0*/ 0, 8, 11, 11, 10, 3, 8, 10, 11, 8, 1, 9, 2, 10, 9, 8, 9, 10,
    /*face1*/ 4, 12, 15, 15, 14, 7, 12, 14, 15, 12, 5, 13, 6, 14, 13, 12, 13,
    14,
    /*face2*/ 0, 11, 16, 16, 15, 4, 11, 15, 16, 3, 19, 11, 7, 15, 19, 11, 19,
    15,
    /*face3*/ 1, 9, 17, 17, 13, 5, 9, 13, 17, 2, 18, 9, 6, 13, 18, 9, 18, 13,
    /*face4*/ 0, 8, 16, 16, 12, 4, 8, 12, 16, 1, 17, 8, 17, 13, 12, 8, 17, 12,
    /*face5*/ 3, 10, 19, 19, 14, 7, 10, 14, 19, 2, 18, 10, 6, 14, 18, 10, 18,
    14,
    // VTK_HEXAHEDRON
    /*face0*/ 0, 1, 2, 0, 2, 3,
    /*face1*/ 4, 5, 6, 4, 6, 7,
    /*face2*/ 0, 3, 4, 3, 7, 4,
    /*face3*/ 1, 2, 5, 2, 6, 5,
    /*face4*/ 0, 1, 4, 1, 5, 4,
    /*face5*/ 3, 2, 7, 2, 6, 7,
    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID
    /*face0*/ 0, 5, 8, 8, 7, 3, 5, 7, 8, 1, 6, 5, 6, 2, 7, 5, 6, 7,
    /*face1*/ 0, 5, 9, 9, 10, 4, 1, 10, 5, 5, 10, 9,
    /*face2*/ 0, 9, 8, 3, 8, 12, 4, 12, 9, 8, 9, 12,
    /*face3*/ 3, 7, 12, 2, 11, 7, 4, 12, 11, 7, 11, 12,
    /*face4*/ 1, 6, 10, 2, 11, 6, 4, 10, 11, 6, 11, 10,
    // VTK_BIQUADRATIC_PYRAMID
    /*face0*/ 0, 5, 8, 8, 7, 3, 5, 7, 8, 1, 6, 5, 6, 2, 7, 5, 6, 7,
    /*face1*/ 0, 5, 9, 9, 10, 4, 1, 10, 5, 5, 10, 9,
    /*face2*/ 0, 9, 8, 3, 8, 12, 4, 12, 9, 8, 9, 12,
    /*face3*/ 3, 7, 12, 2, 11, 7, 4, 12, 11, 7, 11, 12,
    /*face4*/ 1, 6, 10, 2, 11, 6, 4, 10, 11, 6, 11, 10,
    // VTK_PYRAMID
    /*face0*/ 0, 1, 2, 0, 2, 3,
    /*face1*/ 0, 1, 4,
    /*face2*/ 0, 4, 3,
    /*face3*/ 2, 4, 3,
    /*face4*/ 1, 2, 4,
    // VTK_QUAD
    /*face0*/ 0, 1, 2, 0, 2, 3,
    // VTK_QUADRATIC_QUAD
    /*face0*/ 0, 4, 7, 3, 7, 6, 4, 6, 7, 1, 5, 4, 2, 6, 5, 4, 5, 6,
    // VTK_BIQUADRATIC_QUAD
    /*face0*/ 0, 4, 7, 3, 7, 6, 4, 6, 7, 1, 5, 4, 2, 6, 5, 4, 5, 6,
    // VTK_QUADRATIC_TETRA
    /*face0*/ 0, 4, 6, 2, 6, 5, 1, 5, 4, 4, 5, 6,
    /*face1*/ 0, 4, 7, 3, 7, 8, 1, 8, 4, 4, 8, 7,
    /*face2*/ 1, 8, 5, 3, 9, 8, 2, 5, 9, 5, 8, 9,
    /*face3*/ 0, 7, 6, 2, 6, 9, 3, 9, 7, 6, 7, 9,
    // VTK_TETRA
    /*face0*/ 0, 1, 2,
    /*face1*/ 0, 1, 3,
    /*face2*/ 1, 3, 2,
    /*face3*/ 0, 3, 2,
    // VTK_TRIANGLE
    /*face0*/ 0, 1, 2,
    // VTK_QUADRATIC_TRIANGLE
    /*face0*/ 0, 3, 5, 1, 4, 3, 2, 5, 4, 3, 4, 5,
    // VTK_QUADRATIC_WEDGE
    /*face0*/ 0, 6, 8, 1, 7, 6, 2, 8, 7, 6, 7, 8,
    /*face1*/ 3, 9, 11, 4, 10, 9, 5, 11, 10, 9, 10, 11,
    /*face2*/ 0, 6, 12, 12, 9, 3, 6, 9, 12, 1, 13, 6, 4, 9, 13, 6, 13, 9,
    /*face3*/ 1, 7, 13, 4, 13, 10, 7, 10, 13, 2, 14, 7, 5, 10, 14, 7, 14, 10,
    /*face4*/ 0, 8, 12, 12, 11, 3, 8, 11, 12, 2, 14, 8, 5, 11, 14, 8, 14, 11,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    /*face0*/ 0, 6, 8, 1, 7, 6, 2, 8, 7, 6, 7, 8,
    /*face1*/ 3, 9, 11, 4, 10, 9, 5, 11, 10, 9, 10, 11,
    /*face2*/ 0, 6, 12, 12, 9, 3, 6, 9, 12, 1, 13, 6, 4, 9, 13, 6, 13, 9,
    /*face3*/ 1, 7, 13, 4, 13, 10, 7, 10, 13, 2, 14, 7, 5, 10, 14, 7, 14, 10,
    /*face4*/ 0, 8, 12, 12, 11, 3, 8, 11, 12, 2, 14, 8, 5, 11, 14, 8, 14, 11,
    // VTK_WEDGE
    /*face0*/ 0, 1, 2,
    /*face1*/ 3, 4, 5,
    /*face2*/ 0, 1, 4, 0, 4, 3,
    /*face3*/ 1, 2, 4, 2, 5, 4,
    /*face4*/ 0, 2, 3, 2, 5, 3};
inline std::vector<offset_t> getTriFacesOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriFacesOffsets;
  case 2:
    return elemDim2TriFacesOffsets;
  case 3:
    return elemDim3TriFacesOffsets;
  default:
    assert(false);
    return std::vector<offset_t>();
  }
};
inline std::vector<shortint_t> getTriFacesEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriFacesEntries;
  case 2:
    return elemDim2TriFacesEntries;
  case 3:
    return elemDim3TriFacesEntries;
  default:
    assert(false);
    return std::vector<shortint_t>();
  }
};
static std::vector<offset_t> elemDim1TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    3,
    // VTK_QUADRATIC_EDGE
    7,
    // VTK_QUADRATIC_HEXAHEDRON
    7,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    7,
    // VTK_HEXAHEDRON
    7,
    // VTK_VERTEX
    9,
    // VTK_QUADRATIC_PYRAMID
    9,
    // VTK_BIQUADRATIC_PYRAMID
    9,
    // VTK_PYRAMID
    9,
    // VTK_QUAD
    9,
    // VTK_QUADRATIC_QUAD
    9,
    // VTK_BIQUADRATIC_QUAD
    9,
    // VTK_QUADRATIC_TETRA
    9,
    // VTK_TETRA
    9,
    // VTK_TRIANGLE
    9,
    // VTK_QUADRATIC_TRIANGLE
    9,
    // VTK_QUADRATIC_WEDGE
    9,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    9,
    // VTK_WEDGE
    9,
};
static std::vector<shortint_t> elemDim1TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    0, 1, 2,
    // VTK_QUADRATIC_EDGE
    0, 1, 2, 3,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX
    0, 1,
    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD

    // VTK_QUADRATIC_QUAD

    // VTK_BIQUADRATIC_QUAD

    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE

    // VTK_QUADRATIC_TRIANGLE

    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim2TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    2,
    // VTK_QUADRATIC_EDGE
    4,
    // VTK_QUADRATIC_HEXAHEDRON
    4,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    4,
    // VTK_HEXAHEDRON
    4,
    // VTK_VERTEX
    4,
    // VTK_QUADRATIC_PYRAMID
    4,
    // VTK_BIQUADRATIC_PYRAMID
    4,
    // VTK_PYRAMID
    4,
    // VTK_QUAD
    9,
    // VTK_QUADRATIC_QUAD
    14,
    // VTK_BIQUADRATIC_QUAD
    19,
    // VTK_QUADRATIC_TETRA
    19,
    // VTK_TETRA
    19,
    // VTK_TRIANGLE
    23,
    // VTK_QUADRATIC_TRIANGLE
    27,
    // VTK_QUADRATIC_WEDGE
    27,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    27,
    // VTK_WEDGE
    27};
static std::vector<shortint_t> elemDim2TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE
    0, 2,
    // VTK_QUADRATIC_EDGE
    0, 4,
    // VTK_QUADRATIC_HEXAHEDRON

    // VTK_TRIQUADRATIC_HEXAHEDRON

    // VTK_HEXAHEDRON

    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID

    // VTK_BIQUADRATIC_PYRAMID

    // VTK_PYRAMID

    // VTK_QUAD
    0, 2, 4, 6, 8,
    // VTK_QUADRATIC_QUAD
    0, 4, 8, 12, 16,
    // VTK_BIQUADRATIC_QUAD
    0, 4, 8, 12, 16,
    // VTK_QUADRATIC_TETRA

    // VTK_TETRA

    // VTK_TRIANGLE
    0, 2, 4, 6,
    // VTK_QUADRATIC_TRIANGLE
    0, 4, 8, 12,
    // VTK_QUADRATIC_WEDGE

    // VTK_BIQUADRATIC_QUADRATIC_WEDGE

    // VTK_WEDGE

};
static std::vector<offset_t> elemDim3TriLocFacesOffsets{
    // FOR OFFSETING
    0,
    // VTK_EMPTY_CELL
    0,
    // VTK_LINE
    0,
    // VTK_QUADRATIC_EDGE
    0,
    // VTK_QUADRATIC_HEXAHEDRON
    7,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    14,
    // VTK_HEXAHEDRON
    21,
    // VTK_VERTEX
    21,
    // VTK_QUADRATIC_PYRAMID
    27,
    // VTK_BIQUADRATIC_PYRAMID
    33,
    // VTK_PYRAMID
    39,
    // VTK_QUAD
    41,
    // VTK_QUADRATIC_QUAD
    43,
    // VTK_BIQUADRATIC_QUAD
    45,
    // VTK_QUADRATIC_TETRA
    50,
    // VTK_TETRA
    55,
    // VTK_TRIANGLE
    57,
    // VTK_QUADRATIC_TRIANGLE
    59,
    // VTK_QUADRATIC_WEDGE
    65,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    71,
    // VTK_WEDGE
    77};
static std::vector<shortint_t> elemDim3TriLocFacesEntries{
    // VTK_EMPTY_CELL

    // VTK_LINE

    // VTK_QUADRATIC_EDGE

    // VTK_QUADRATIC_HEXAHEDRON
    0, 18, 36, 54, 72, 90, 108,
    // VTK_TRIQUADRATIC_HEXAHEDRON
    0, 18, 36, 54, 72, 90, 108,
    // VTK_HEXAHEDRON
    0, 6, 12, 18, 24, 30, 36,
    // VTK_VERTEX

    // VTK_QUADRATIC_PYRAMID
    0, 18, 30, 42, 54, 66,
    // VTK_BIQUADRATIC_PYRAMID
    0, 18, 30, 42, 54, 66,
    // VTK_PYRAMID
    0, 6, 9, 12, 15, 18,
    // VTK_QUAD
    0, 6,
    // VTK_QUADRATIC_QUAD
    0, 18,
    // VTK_BIQUADRATIC_QUAD
    0, 18,
    // VTK_QUADRATIC_TETRA
    0, 12, 24, 36, 48,
    // VTK_TETRA
    0, 3, 6, 9, 12,
    // VTK_TRIANGLE
    0, 3,
    // VTK_QUADRATIC_TRIANGLE
    0, 12,
    // VTK_QUADRATIC_WEDGE
    0, 12, 24, 42, 60, 78,
    // VTK_BIQUADRATIC_QUADRATIC_WEDGE
    0, 12, 24, 42, 60, 78,
    // VTK_WEDGE
    0, 3, 6, 12, 18, 24};
inline std::vector<offset_t> getTriLocFacesOffsets(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriLocFacesOffsets;
  case 2:
    return elemDim2TriLocFacesOffsets;
  case 3:
    return elemDim3TriLocFacesOffsets;
  default:
    assert(false);
    return std::vector<offset_t>();
  }
};
inline std::vector<shortint_t> getTriLocFacesEntries(const int_t dimension) {
  switch (dimension) {
  case 1:
    return elemDim1TriLocFacesEntries;
  case 2:
    return elemDim2TriLocFacesEntries;
  case 3:
    return elemDim3TriLocFacesEntries;
  default:
    assert(false);
    return std::vector<shortint_t>();
  }
};

}; // namespace FiniteElement
} // namespace PACMAN
