//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#include "interface.hpp"

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <stdexcept>
#include <string>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
// Extern-template declarations in interpolator.hxx suppress re-instantiation
// in this TU. ETI is provided by MLS_interface_eti_*.cpp.
#include "mls/interpolator.hxx"

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

using namespace PACMAN;

template <typename ExecSpace, PACMAN::int_t Dim>
PACMAN::MLSInterpolateResult
RunMLSInterpolate(PACMAN::coordinates_t *pSourcePoints,
                  PACMAN::int_t nSourcePoints, PACMAN::fp_t *pSourceValues,
                  PACMAN::coordinates_t *pTargetPoints,
                  PACMAN::int_t nTargetPoints) {
  PACMAN::Transfer<ExecSpace, Dim> transfer(PACMAN::TransferMethods::MLS);
  PACMAN::SetupTransferClass(transfer, nSourcePoints, 0, 0, nTargetPoints,
                             pSourcePoints, pSourceValues, nullptr, nullptr,
                             nullptr, pTargetPoints);

  PACMAN::MLS::MLSInterpolator<ExecSpace, Dim> interpolator(transfer);

  PACMAN::MLSInterpolateResult result;
  result.targetValues.resize(nTargetPoints);

  auto unmanaged_host_tv =
      Kokkos::View<PACMAN::fp_t *,
                   Kokkos::DefaultHostExecutionSpace::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          result.targetValues.data(), nTargetPoints);
  Kokkos::deep_copy(unmanaged_host_tv, transfer.targetValues);
  return result;
}

template <PACMAN::int_t Dim>
requires PACMAN::IsValidDim<Dim> PACMAN::MLSInterpolateResult
DispatchMLS(unsigned char execSpace,
            PACMAN::coordinates_t *pSourcePoints, PACMAN::int_t nSourcePoints,
            PACMAN::fp_t *pSourceValues, PACMAN::coordinates_t *pTargetPoints,
            PACMAN::int_t nTargetPoints) {
  return std::visit(
      [&](auto execSpaceObj) {
        return RunMLSInterpolate<decltype(execSpaceObj), Dim>(
            pSourcePoints, nSourcePoints, pSourceValues, pTargetPoints,
            nTargetPoints);
      },
      PACMAN::MakeExecSpace(execSpace));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

namespace PACMAN {

MLSInterpolateResult
MLS_interpolate(int_t spaceDimension, unsigned char execSpace,
                coordinates_t *sourcePoints, int_t nSourcePoints,
                fp_t *sourceValues, coordinates_t *targetPoints,
                int_t nTargetPoints) {
  const std::string _region_name = "PACMAN::MLS_interpolate";
  const Kokkos::Profiling::ScopedRegion region(_region_name);

  if (sourcePoints == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourcePoints is null but nSourcePoints > 0");
  if (targetPoints == nullptr && nTargetPoints > 0)
    throw std::invalid_argument("targetPoints is null but nTargetPoints > 0");
  if (sourceValues == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourceValues is null but nSourcePoints > 0");

  switch (spaceDimension) {
  case 1:
    return DispatchMLS<1>(execSpace, sourcePoints, nSourcePoints, sourceValues,
                          targetPoints, nTargetPoints);
  case 2:
    return DispatchMLS<2>(execSpace, sourcePoints, nSourcePoints, sourceValues,
                          targetPoints, nTargetPoints);
  case 3:
    return DispatchMLS<3>(execSpace, sourcePoints, nSourcePoints, sourceValues,
                          targetPoints, nTargetPoints);
  default:
    throw std::runtime_error(
        "The dimension of the points can only be: 1, 2 or 3.\n");
  }
}

} // namespace PACMAN
