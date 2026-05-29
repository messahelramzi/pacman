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
// in this TU. ETI is provided by rbf_interface_eti_*.cpp.
#include "rbf_pum/interpolator.hxx"
#include "solver/rbf_functions.hpp"

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

using namespace PACMAN;

using AvailableRbfFunctions =
    std::variant<RbfPum::WendlandC0, RbfPum::WendlandC2, RbfPum::WendlandC4,
                 RbfPum::WendlandC6, RbfPum::WendlandC8>;

static AvailableRbfFunctions MakeRbfFunction(unsigned char rbfFunction) {
  switch (rbfFunction) {
  case RbfFunctions::WENDLANDC0:
    return RbfPum::WendlandC0{};
  case RbfFunctions::WENDLANDC2:
    return RbfPum::WendlandC2{};
  case RbfFunctions::WENDLANDC4:
    return RbfPum::WendlandC4{};
  case RbfFunctions::WENDLANDC6:
    return RbfPum::WendlandC6{};
  case RbfFunctions::WENDLANDC8:
    return RbfPum::WendlandC8{};
  default:
    throw std::invalid_argument("Unsupported RBF function selector: " +
                                std::to_string(rbfFunction));
  }
}

template <typename ExecSpace, typename RbfFuncType, PACMAN::int_t Dim>
PACMAN::RbfInterpolateResult
RunRbfInterpolate(PACMAN::coordinates_t *pSourcePoints,
                  PACMAN::int_t nSourcePoints, PACMAN::fp_t *pSourceValues,
                  PACMAN::coordinates_t *pTargetPoints,
                  PACMAN::int_t nTargetPoints) {
  PACMAN::Transfer<ExecSpace, Dim> transfer(PACMAN::TransferMethods::RBF_PUM);
  PACMAN::SetupTransferClass(transfer, nSourcePoints, 0, 0, nTargetPoints,
                             pSourcePoints, pSourceValues, nullptr, nullptr,
                             nullptr, pTargetPoints);

  PACMAN::RbfPum::RbfPumInterpolator<ExecSpace, Dim, RbfFuncType> interpolator(
      transfer);

  PACMAN::RbfInterpolateResult result;
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
requires PACMAN::IsValidDim<Dim> PACMAN::RbfInterpolateResult
DispatchRbf(unsigned char execSpace, unsigned char rbfFunction,
            PACMAN::coordinates_t *pSourcePoints, PACMAN::int_t nSourcePoints,
            PACMAN::fp_t *pSourceValues, PACMAN::coordinates_t *pTargetPoints,
            PACMAN::int_t nTargetPoints) {
  return std::visit(
      [&](auto execSpaceObj, auto rbfFunctionObj) {
        return RunRbfInterpolate<decltype(execSpaceObj),
                                 decltype(rbfFunctionObj), Dim>(
            pSourcePoints, nSourcePoints, pSourceValues, pTargetPoints,
            nTargetPoints);
      },
      PACMAN::MakeExecSpace(execSpace), MakeRbfFunction(rbfFunction));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

namespace PACMAN {

RbfInterpolateResult
rbf_interpolate(int_t spaceDimension, unsigned char execSpace,
                unsigned char rbfFunction, coordinates_t *sourcePoints,
                int_t nSourcePoints, fp_t *sourceValues,
                coordinates_t *targetPoints, int_t nTargetPoints) {
  const std::string _region_name = "PACMAN::rbf_interpolate";
  const Kokkos::Profiling::ScopedRegion region(_region_name);

  if (sourcePoints == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourcePoints is null but nSourcePoints > 0");
  if (targetPoints == nullptr && nTargetPoints > 0)
    throw std::invalid_argument("targetPoints is null but nTargetPoints > 0");
  if (sourceValues == nullptr && nSourcePoints > 0)
    throw std::invalid_argument("sourceValues is null but nSourcePoints > 0");

  switch (spaceDimension) {
  case 1:
    return DispatchRbf<1>(execSpace, rbfFunction, sourcePoints, nSourcePoints,
                          sourceValues, targetPoints, nTargetPoints);
  case 2:
    return DispatchRbf<2>(execSpace, rbfFunction, sourcePoints, nSourcePoints,
                          sourceValues, targetPoints, nTargetPoints);
  case 3:
    return DispatchRbf<3>(execSpace, rbfFunction, sourcePoints, nSourcePoints,
                          sourceValues, targetPoints, nTargetPoints);
  default:
    throw std::runtime_error(
        "The dimension of the points can only be: 1, 2 or 3.\n");
  }
}

} // namespace PACMAN
