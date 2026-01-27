#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_serial_gemv.hpp>
// #include <KokkosBatched_LU_Decl.hpp>
// #include <KokkosBatched_Trsv_Decl.hpp>

#include "finite_elements/utils/Algebra.hpp"
#include "finite_elements/utils/FETools.hxx"
namespace PACMAN {
namespace FiniteElements {

/** Tolerance to stop the newton, dxietaphi.squaredNorm()*/
#define D_XI_ETA_PHI_TOL_SQUARED 1e-10
/** Maximal number of newton iters */
#define MAX_NEWTON_ITER 15

template <typename ExecSpace, typename FEspace, int_t Dim>
/**
 * @brief The Newton method applied to a finite element. The targetPoint is
 * updated to hold the barycentric coordinates ultimately.
 * @param[in] Xcoor The coordinates for each of the node of the element.
 * @param[in] targetPoint The coordinates of the point in space we are looking
 * for its barycentric coordinate.
 * @param[out] weights Evaluation of the shape function at the barycentric
 * coordinate.
 * @return true if the target point is contained in the finite element.
 */
KOKKOS_FUNCTION bool
ApplyNewton(const Kokkos::View<coordinates_t **, ExecSpace> Xcoor,
            const Kokkos::View<coordinates_t[Dim], ExecSpace> targetPoint,
            Kokkos::View<fp_t *, ExecSpace> weights,
            const bool forceEvaluation = false) {

  Kokkos::Array<fp_t, 3> xietaphi_arr;
  Kokkos::View<fp_t *, ExecSpace> xietaphi(xietaphi_arr.data(), Dim);
  for (int j = 0; j < Dim; j++) {
    xietaphi(j) = 0.5;
  }
  KOKKOS_ASSERT(Xcoor.extent_int(0) == FEspace::numberOfShapeFunctions);
  // Shape function values at the current xi,eta,phi
  Kokkos::Array<fp_t, FEspace::numberOfShapeFunctions> sfv_arr;
  Kokkos::View<fp_t *, ExecSpace> sfv(sfv_arr.data(),
                                      FEspace::numberOfShapeFunctions);
  // Coordinates in the global system
  Kokkos::Array<fp_t, Dim> f_arr;
  Kokkos::View<fp_t *, ExecSpace> f(f_arr.data(), Dim);
  // Derivatives of shape functions at the current xi,eta,phi
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::numberOfShapeFunctions>
      dN_arr;
  Kokkos::View<fp_t **, ExecSpace> dN(dN_arr.data(), FEspace::dimensionality,
                                      FEspace::numberOfShapeFunctions);
  // Derivative of f w.r.t xi,eta,phi
  Kokkos::Array<fp_t, FEspace::dimensionality * Dim> dNXcoor_arr;
  Kokkos::View<fp_t **, ExecSpace> dNXcoor(dNXcoor_arr.data(),
                                           FEspace::dimensionality, Dim);
  // Numerical derivative of f w.r.t xi,eta,phi
  Kokkos::Array<fp_t, FEspace::dimensionality> dfNum_arr;
  Kokkos::View<fp_t *, ExecSpace> dfNum(dfNum_arr.data(),
                                        FEspace::dimensionality);
  // Newton function to solve h * dxietaphi = dfNum
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::dimensionality> h_arr;
  Kokkos::View<fp_t **, ExecSpace> h(h_arr.data(), FEspace::dimensionality,
                                     FEspace::dimensionality);
  // temporary arrays for second derivative computations
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::dimensionality>
      tmp_arr;
  Kokkos::View<fp_t **, ExecSpace> tmp(tmp_arr.data(), FEspace::dimensionality,
                                       FEspace::dimensionality);
  Kokkos::Array<fp_t, FEspace::dimensionality * FEspace::dimensionality>
      ddNi_arr;
  Kokkos::View<fp_t **, ExecSpace> ddNi(
      ddNi_arr.data(), FEspace::dimensionality, FEspace::dimensionality);
  // Newton update dxietaphi
  Kokkos::Array<fp_t, FEspace::dimensionality> dxietaphi_arr;
  Kokkos::View<fp_t *, ExecSpace> dxietaphi(dxietaphi_arr.data(),
                                            FEspace::dimensionality);

  FEspace::UpdateShapeFunctionsValues(xietaphi_arr[0], xietaphi_arr[1],
                                      xietaphi_arr[2], sfv);
  KokkosBlas::Experimental::serial_gemv('T', fp_consts::one(), Xcoor, sfv,
                                        fp_consts::zero(),
                                        f); // f= sfv.transpose() * xcoor;
  KokkosBlas::serial_axpy(-fp_consts::one(), targetPoint,
                          f); // f -= targetPoint;

  int_t cnt = 0;
  using namespace KokkosBatched;

  for (; cnt < MAX_NEWTON_ITER; ++cnt) {
    KokkosBlas::Experimental::serial_gemv('T', fp_consts::one(), Xcoor, sfv,
                                          fp_consts::zero(),
                                          f); // f= sfv.transpose() * xcoor;
    KokkosBlas::serial_axpy(-fp_consts::one(), targetPoint,
                            f); // f -= targetPoint;

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
      for (int_t j = 0; j < Dim; j++) { // cols
        for (int_t k = 0; k < FEspace::dimensionality * FEspace::dimensionality;
             k++) {
          tmp_arr[k] = 0.0;
        }
        for (int_t i = 0; i < FEspace::numberOfShapeFunctions; i++) { // rows
          for (int_t k = 0;
               k < FEspace::dimensionality * FEspace::dimensionality; k++) {
            ddNi_arr[k] = 0.0;
          }
          FEspace::UpdateShapeFunctionsDerDerValues(
              i, xietaphi_arr[0], xietaphi_arr[1], xietaphi_arr[2], ddNi);
          KokkosBlas::serial_axpy(Xcoor(i, j), ddNi, tmp);
        }
        KokkosBlas::serial_axpy(f(j), tmp, h);
      }
    }

    // Here we use a LU decomposition + backsubstitution dxietaphi = Inv(h) *
    // dfNum dfNum inplace reused to store the result od dxietaphi
    // SerialLU<Algo::LU::Unblocked>
    //    ::invoke(h);
    // SerialTrsv<Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsv::Unblocked>
    //    ::invoke(one, h, dfNum);
    FiniteElements::Matrix<ExecSpace, FEspace::dimensionality>::InvInPlace(h);
    KokkosBlas::Experimental::serial_gemv('N', fp_consts::one(), h, dfNum,
                                          fp_consts::zero(), dxietaphi);

    for (int_t j = 0; j < FEspace::dimensionality; j++) {
      xietaphi(j) -= dxietaphi(j);
    }

    FEspace::UpdateShapeFunctionsValues(xietaphi_arr[0], xietaphi_arr[1],
                                        xietaphi_arr[2], sfv);
    KokkosBlas::Experimental::serial_gemv('T', fp_consts::one(), Xcoor, sfv,
                                          fp_consts::zero(),
                                          f); // f= sfv.transpose() * xcoor;
    KokkosBlas::serial_axpy(-fp_consts::one(), targetPoint,
                            f); // f -= targetPoint;

    fp_t norm2 = fp_consts::zero();
    fp_t scale = fp_consts::zero();
    for (int_t j = 0; j < Dim; j++) {
      norm2 += f(j) * f(j);
      scale += targetPoint(j) * targetPoint(j);
    }
    if (norm2 / scale < D_XI_ETA_PHI_TOL_SQUARED) {
      break;
    }
  }

  if (forceEvaluation) {
    for (int_t i = 0; i < FEspace::numberOfShapeFunctions; ++i) {
      weights(i) = sfv(i);
    }
    return true;
  }

  if ((cnt < MAX_NEWTON_ITER) &&
      FEspace::isInside(xietaphi_arr[0], xietaphi_arr[1], xietaphi_arr[2])) {
    for (int_t i = 0; i < FEspace::numberOfShapeFunctions; ++i) {
      weights(i) = sfv(i);
    }
    return true;
  }
  return false;
}
} // namespace FiniteElements
} // namespace PACMAN
