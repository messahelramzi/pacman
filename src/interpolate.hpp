#pragma once

#include <iostream>
#include <utility>

#include "common/transfer.hxx"
#include "finite_elements/FTInterpClamp.hpp"
#include "finite_elements/FTInterpExtrap.hpp"
#include "finite_elements/FTInterpNearest.hpp"
#include "finite_elements/FTInterpZero.hpp"
#include "finite_elements/FTNearest.hpp"
#include "rbf_pum/interpolator.hpp"

namespace PACMAN {
template <typename ExecSpace, int_t Dim>
void Interpolate(Transfer<ExecSpace, Dim> &transfer) {
  const TransferMethods method = transfer.method;

  switch (method) {
  case TransferMethods::NEAREST_NEAREST:
    FiniteElements::FTNearest(transfer);
    break;
  case TransferMethods::INTERP_NEAREST:
    FiniteElements::FTInterpNearest(transfer);
    break;
  case TransferMethods::INTERP_ZEROFILL:
    FiniteElements::FTInterZero(transfer);
    break;
  case TransferMethods::INTERP_EXTRAP:
    FiniteElements::FTInterExtrap(transfer);
    break;
  case TransferMethods::INTERP_CLAMP:
    FiniteElements::FTInterClamp(transfer);
    break;
  case TransferMethods::RBF_PUM: {
    RbfPum::RbfPumInterpolator<ExecSpace, Dim, RbfPum::WendlandC6>
        rbf_interpolator(transfer);
    break;
  }
  default:
    std::cerr << "Invalid transfer method" << std::endl;
    assert(false);
    break;
  }
  return;
}
} // namespace PACMAN
