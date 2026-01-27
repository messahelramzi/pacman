//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/ComputeLinearSkin.hxx"
#include "finite_elements/utils/FTUtils.hpp"
#include "finite_elements/utils/SkinProjection.hpp"

namespace PACMAN {

namespace FiniteElements {

template <typename ExecSpace, int_t Dim>
/**
 * @brief Compute the target point finite element interpolation if intersects or
 * assign nearest element interpolation values after projection of the target
 * point if outside all elements. * @param[in] transfer the transfer class
 * holding informations.
 * @return target point values
 */
void FTInterClamp(Transfer<ExecSpace, Dim> &transfer) {
  Kokkos::Profiling::pushRegion("FiniteElement::FTInterpClamp");
  Kokkos::Profiling::pushRegion("Compute target point FE intersection");

  ComputeBoxTargetPointIntersection<ExecSpace, Dim>(transfer);

  Kokkos::fence();

  Kokkos::Profiling::popRegion(); // Compute target point FE intersection

  Kokkos::Profiling::pushRegion(
      "Compute target point projection on triangulated skin mesh");
  ComputeLinearSkin<ExecSpace, Dim>(transfer);
  if constexpr (Dim == 1) {
    ComputeProjectionOn1DSkin<ExecSpace>(transfer, false);
  } else if constexpr (Dim == 2) {
    ComputeProjectionOn2DSkin<ExecSpace>(transfer, false);
  } else if constexpr (Dim == 3) {
    ComputeProjectionOn3DSkin<ExecSpace>(transfer, false);
  } else {
    throw std::runtime_error("Unexpected dimension for projection");
  }
  Kokkos::fence();

  Kokkos::Profiling::popRegion(); // Compute target point projection on
                                  // triangulated skin mesh
  Kokkos::Profiling::popRegion(); // FTInterpClamp
}
} // namespace FiniteElements
} // namespace PACMAN
