//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/ComputeLinearSkin.hpp"
#include "finite_elements/utils/FTUtils.hpp"
#include "finite_elements/utils/SkinProjection.hpp"

namespace PACMAN {

namespace FiniteElements {

/**
 * @brief Interpolate target points from FE cells and extrapolate outside points
 * via projection onto the source mesh skin to retrieve FE cell for
 * extrapolation.
 * @tparam ExecSpace Kokkos execution space used for search/projection kernels.
 * @tparam Dim Spatial dimension of the interpolation problem.
 * @param[in,out] transfer Transfer descriptor holding interpolation data.
 * Reads: source mesh/field data and target points.
 * Writes: `targetValues` and `targetStatus`.
 * @note Outside points are projected on the linearized boundary skin before
 * interpolation/extrapolation.
 */
template <typename ExecSpace, int_t Dim>
void FTInterExtrap(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion("FiniteElement::FTInterpExtrap");

  Kokkos::Profiling::pushRegion("Compute target point FE intersection");
  ComputeBoxTargetPointIntersection<ExecSpace, Dim>(transfer);

  Kokkos::Profiling::popRegion(); // Compute target point FE intersection

  Kokkos::Profiling::pushRegion(
      "Compute target point projection on triangulated skin mesh");
  ComputeLinearSkin<ExecSpace, Dim>(transfer);
  if constexpr (Dim == 1) {
    ComputeProjectionOn1DSkin<ExecSpace>(transfer, true);
  } else if constexpr (Dim == 2) {
    ComputeProjectionOn2DSkin<ExecSpace>(transfer, true);
  } else if constexpr (Dim == 3) {
    ComputeProjectionOn3DSkin<ExecSpace>(transfer, true);
  } else {
    throw std::runtime_error("Unexpected dimension for projection");
  }
  Kokkos::Profiling::popRegion(); // Compute target point projection on
                                  // triangulated skin mesh

  Kokkos::Profiling::popRegion(); // FTInterpExtrap
}
} // namespace FiniteElements
} // namespace PACMAN
