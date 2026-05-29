//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas2_serial_gemv.hpp>

#include "finite_elements/utils/FETools.hpp"

namespace PACMAN {
namespace FiniteElements {

/** Tolerance to stop the newton, dxietaphi.squaredNorm()*/
#define D_XI_ETA_PHI_TOL_SQUARED 1e-10
/** Maximal number of newton iters */
#define MAX_NEWTON_ITER 15

#define CaseSwitch(CellType_STRUCT)                                            \
  case CellType_STRUCT: {                                                      \
    return ApplyNewton<ExecSpace,                                              \
                       LagrangeSpaceGeo<ExecSpace, CellType_STRUCT>, Dim>(     \
        Xcoor, targetPoint, weights, forceEvaluation);                         \
    break;                                                                     \
  }

/**
 * @brief Solve FE inverse mapping with Newton iterations and compute
 * interpolation weights.
 * @tparam ExecSpace Kokkos execution space used for local views/operations.
 * @tparam FEspace Finite-element type providing shape functions and geometry
 * predicates.
 * @tparam Dim Spatial dimension of the embedding space.
 * @param[in] Xcoor Coordinates of the current element nodes.
 * @param[in] targetPoint Coordinates of the query point in physical space.
 * @param[out] weights Shape-function values evaluated at the solved local
 * coordinates.
 * @param[in] forceEvaluation If `true`, always write `weights` and return
 * `true` even when the point is outside the reference element.
 * @return `true` if Newton converges and the point is inside the element,
 * or if `forceEvaluation` is enabled; otherwise `false`.
 */
template <typename ExecSpace, typename FEspace, int_t Dim>
KOKKOS_FUNCTION bool
ApplyNewton(const Kokkos::View<coordinates_t **, ExecSpace> Xcoor,
            const Kokkos::View<coordinates_t[Dim], ExecSpace> targetPoint,
            Kokkos::View<fp_t *, ExecSpace> weights,
            const bool forceEvaluation = false) {

  // Shape function values at the current xi,eta,phi stored in weights
  // KOKKOS_ASSERT(Xcoor.extent_int(0) == FEspace::numberOfShapeFunctions);

  Kokkos::Array<fp_t, 3> xietaphi_arr = {0.5, 0.5, 0.5};
  Kokkos::View<fp_t *, ExecSpace> xietaphi(xietaphi_arr.data(), Dim);

  // Coordinates in the global system [DIM]
  Kokkos::Array<fp_t, Dim> f_arr;
  Kokkos::View<fp_t *, ExecSpace> f(f_arr.data(), Dim);
  // Derivatives of shape functions at the current xi,eta,phi [FE::DIM;
  // FE::NSHAPES]
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::numberOfShapeFunctions>
      dN_arr;
  Kokkos::View<fp_t **, ExecSpace> dN(dN_arr.data(), FEspace::dimensionality,
                                      FEspace::numberOfShapeFunctions);
  // Derivative of f w.r.t xi,eta,phi [FE::DIM; DIM]
  Kokkos::Array<fp_t, FEspace::dimensionality * Dim> dNXcoor_arr;
  Kokkos::View<fp_t **, ExecSpace> dNXcoor(dNXcoor_arr.data(),
                                           FEspace::dimensionality, Dim);
  // Numerical derivative of f w.r.t xi,eta,phi [FE::DIM]
  Kokkos::Array<fp_t, FEspace::dimensionality> dfNum_arr;
  Kokkos::View<fp_t *, ExecSpace> dfNum(dfNum_arr.data(),
                                        FEspace::dimensionality);
  // Newton function to solve h * dxietaphi = dfNum [FE::DIM, FE::DIM]
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::dimensionality> h_arr;
  Kokkos::View<fp_t **, ExecSpace> h(h_arr.data(), FEspace::dimensionality,
                                     FEspace::dimensionality);

  FEspace::UpdateShapeFunctionsValues(xietaphi_arr[0], xietaphi_arr[1],
                                      xietaphi_arr[2], weights);
  // f= weights.transpose() * xcoor;
  KokkosBlas::Experimental::serial_gemv('T', fp_consts::one(), Xcoor, weights,
                                        fp_consts::zero(), f);
  // f -= targetPoint;
  KokkosBlas::serial_axpy(-fp_consts::one(), targetPoint, f);

  int_t cnt = 0;
  double error = 0.0;

  using namespace KokkosBatched;

  for (; cnt < MAX_NEWTON_ITER; ++cnt) {

    FEspace::UpdateShapeFunctionsDerValues(xietaphi_arr[0], xietaphi_arr[1],
                                           xietaphi_arr[2], dN);

    // dfNum = (f * ((dN * xcoor).transpose())).transpose();
    // here h = (dN * xcoor)
    SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
               Algo::Gemm::Unblocked>::invoke(fp_consts::one(), dN, Xcoor,
                                              fp_consts::zero(), dNXcoor);
    KokkosBlas::Experimental::serial_gemv('N', fp_consts::one(), dNXcoor, f,
                                          fp_consts::zero(), dfNum);

    // h  = (dN * xcoor) * (dN * xcoor).transpose();
    SerialGemm<Trans::NoTranspose, Trans::Transpose,
               Algo::Gemm::Unblocked>::invoke(fp_consts::one(), dNXcoor,
                                              dNXcoor, fp_consts::zero(), h);

    if (cnt > 9) {
      Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::dimensionality>
          ddNi_arr;
      Kokkos::View<fp_t **, ExecSpace> ddNi(
          ddNi_arr.data(), FEspace::dimensionality, FEspace::dimensionality);
      for (int_t j = 0; j < Dim; j++) {                               // cols
        for (int_t i = 0; i < FEspace::numberOfShapeFunctions; i++) { // rows
          for (int_t k = 0;
               k < FEspace::dimensionality * FEspace::dimensionality; k++) {
            ddNi_arr[k] = 0.0;
          }
          FEspace::UpdateShapeFunctionsDerDerValues(
              i, xietaphi_arr[0], xietaphi_arr[1], xietaphi_arr[2], ddNi);
          KokkosBlas::serial_axpy(Xcoor(i, j), ddNi, h);
        }
      }
    }

    SerialLU<Algo::LU::Unblocked>::invoke(h);
    SerialTrsv<Uplo::Lower, Trans::NoTranspose, Diag::Unit,
               Algo::Trsv::Unblocked>::invoke(fp_consts::one(), h, dfNum);
    SerialTrsv<Uplo::Upper, Trans::NoTranspose, Diag::NonUnit,
               Algo::Trsv::Unblocked>::invoke(fp_consts::one(), h, dfNum);

    for (int_t j = 0; j < FEspace::dimensionality; j++) {
      xietaphi(j) -= dfNum(j);
    }

    FEspace::UpdateShapeFunctionsValues(xietaphi_arr[0], xietaphi_arr[1],
                                        xietaphi_arr[2], weights);
    KokkosBlas::Experimental::serial_gemv('T', fp_consts::one(), Xcoor, weights,
                                          fp_consts::zero(),
                                          f); // f= weights.transpose() * xcoor;
    KokkosBlas::serial_axpy(-fp_consts::one(), targetPoint,
                            f); // f -= targetPoint;

    // const fp_t error =  KokkosBlas::dot(f, f) / KokkosBlas::dot(targetPoint,
    // targetPoint);
    fp_t norm2 = fp_consts::zero();
    fp_t scale = fp_consts::zero();
    for (int_t j = 0; j < Dim; j++) {
      norm2 += f(j) * f(j);
      scale += targetPoint(j) * targetPoint(j);
    }
    if (error < D_XI_ETA_PHI_TOL_SQUARED) {
      break;
    }
  }

  if (forceEvaluation) {
    return true;
  }

  if ((cnt < MAX_NEWTON_ITER) &&
      FEspace::isInside(xietaphi_arr[0], xietaphi_arr[1], xietaphi_arr[2])) {
    return true;
  }

  return false;
}

/**
 * @brief Dispatch Newton-based FE inversion to the correct element type.
 * @tparam ExecSpace Kokkos execution space used for local views/operations.
 * @tparam Dim Spatial dimension of the interpolation problem.
 * @param[in] type PACMAN cell type identifier of the current element.
 * @param[in] Xcoor Coordinates of the current element nodes.
 * @param[in] targetPoint Query point coordinates.
 * @param[out] weights Interpolation weights (shape-function values).
 * @param[in] forceEvaluation If `true`, evaluate weights even for points
 * outside the reference element.
 * @return `true` if the point is considered valid for interpolation for the
 * selected element policy; otherwise `false`.
 */
template <typename ExecSpace, int_t Dim>
KOKKOS_FUNCTION bool ApplyNewtonOnElement(
    const cell_t type, const Kokkos::View<coordinates_t **, ExecSpace> Xcoor,
    const Kokkos::View<coordinates_t[Dim], ExecSpace> targetPoint,
    Kokkos::View<fp_t *, ExecSpace> weights,
    const bool forceEvaluation = false) {
  // clang-format off
  switch (static_cast<CellType>(type)) {
    CaseSwitch(CellType::VTK_LINE)
    CaseSwitch(CellType::VTK_QUADRATIC_HEXAHEDRON)
    CaseSwitch(CellType::VTK_QUADRATIC_EDGE)
    CaseSwitch(CellType::VTK_HEXAHEDRON)
    CaseSwitch(CellType::VTK_QUADRATIC_PYRAMID)
    CaseSwitch(CellType::VTK_PYRAMID)
    CaseSwitch(CellType::VTK_QUAD)
    CaseSwitch(CellType::VTK_QUADRATIC_QUAD)
    CaseSwitch(CellType::VTK_QUADRATIC_TETRA)
    CaseSwitch(CellType::VTK_TETRA)
    CaseSwitch(CellType::VTK_TRIANGLE)
    CaseSwitch(CellType::VTK_QUADRATIC_TRIANGLE)
    CaseSwitch(CellType::VTK_QUADRATIC_WEDGE)
    CaseSwitch(CellType::VTK_WEDGE)
    default:
      char errorMsg[] = "Unexpected element type XXX";
      errorMsg[sizeof(errorMsg) - 4] = '0' + ((int)type / 100);
      errorMsg[sizeof(errorMsg) - 3] = '0' + (((int)type / 10) % 10);
      errorMsg[sizeof(errorMsg) - 2] = '0' + ((int)type % 10);
      Kokkos::abort(errorMsg);
      return false;
  }
  // clang-format on
  return false;
}
} // namespace FiniteElements
} // namespace PACMAN
