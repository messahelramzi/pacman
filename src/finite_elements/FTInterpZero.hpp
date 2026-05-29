//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FTUtils.hpp"
#include <Kokkos_Core.hpp>

namespace PACMAN {

namespace FiniteElements {

/**
 * @brief Interpolate target points from FE cells, zero-filling points outside
 * the source mesh.
 * @tparam ExecSpace Kokkos execution space used for the intersection/query
 * kernels.
 * @tparam Dim Spatial dimension of the interpolation problem.
 * @param[in,out] transfer Transfer descriptor holding interpolation data.
 * Reads: source mesh/field data and target points.
 * Writes: `targetValues` and `targetStatus`.
 * @note Points that do not intersect any source element are assigned a zero
 * value.
 */
template <typename ExecSpace, int Dim>
void FTInterZero(Transfer<ExecSpace, Dim> &transfer) {

  Kokkos::Profiling::pushRegion("FiniteElement::FTInterpZero");

  Kokkos::Profiling::pushRegion("Compute target point FE intersection");
  ComputeBoxTargetPointIntersection<ExecSpace, Dim>(transfer);
  Kokkos::Profiling::popRegion(); // Compute target point FE intersection

  Kokkos::Profiling::popRegion(); // FTInterpZero
}
} // namespace FiniteElements
} // namespace PACMAN
