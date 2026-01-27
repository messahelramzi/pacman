//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

#pragma once

#include "finite_elements/utils/ArborXCallbacks.hpp"
#include "finite_elements/utils/FTUtils.hpp"
#include <Kokkos_Core.hpp>

namespace PACMAN {

namespace FiniteElements {

template <typename ExecSpace, int Dim>
/**
 * @brief Compute the target point finite element interpolation if intersects or
 * assign zero node value if outside all elements.
 * @param[in] transfer the transfer class holding informations.
 * @return target point values
 */
void FTInterZero(Transfer<ExecSpace, Dim> &transfer) {

  Kokkos::Profiling::pushRegion("FiniteElement::FTInterpZero");

  Kokkos::Profiling::pushRegion("Compute target point FE intersection");
  ComputeBoxTargetPointIntersection<ExecSpace, Dim>(transfer);
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // Compute target point FE intersection

  Kokkos::Profiling::popRegion(); // FTInterpZero
}
} // namespace FiniteElements
} // namespace PACMAN
