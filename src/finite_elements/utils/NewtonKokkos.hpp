#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas2_serial_gemv.hpp>

#include "finite_elements/utils/Algebra.hpp"
#include "finite_elements/utils/FETools.hxx"

/** Tolerance to stop the newton,  dxietaphi.squaredNorm()*/
#define D_XI_ETA_PHI_TOL_SQUARED 1e-10
/** Maximal number of newton iters */
#define MAX_NEWTON_ITER 25

namespace PACMAN {
namespace FiniteElements {
template <CellType CT, typename A, typename C, typename D>
KOKKOS_INLINE_FUNCTION static void /*Eigen::Matrix<fp_t,
                                      geoStruct::dimensionality,
                                      geoStruct::dimensionality>*/
DdfPart2(const A f, const C & /*dN*/,
         const D &coordAtDofs /*, const MatrixD31& xietaphi*/) {
  // Eigen::Matrix<fp_t, geoStruct::dimensionality, geoStruct::dimensionality>
  // res; res.fill(0.0); Eigen::Matrix<fp_t, geoStruct::dimensionality,
  // geoStruct::dimensionality> temp; for (int i = 0; i < coordAtDofs.cols();
  // ++i) {
  //     temp.fill(0.0);
  //     for (int j = 0; j < coordAtDofs.rows(); ++j) {
  //         Eigen::Matrix<fp_t, geoStruct::dimensionality,
  //         geoStruct::dimensionality> ddNi; ddNi.setZero();
  //         geoStruct::UpdateShapeFunctionsDerDerValues(j, xietaphi(0, 0),
  //         xietaphi(1, 0), xietaphi(2, 0), ddNi); temp += ddNi *
  //         coordAtDofs(j, i);
  //     }
  //     res -= f(0, i) * temp;
  // }
  // return res;
  return;
}

template <typename ExecSpace, typename FEspace, unsigned int spaceDimension>
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
KOKKOS_FUNCTION bool ApplyNewton(
    const Kokkos::View<coordinates_t **, ExecSpace> Xcoor,
    const Kokkos::View<coordinates_t[spaceDimension], ExecSpace> targetPoint,
    Kokkos::View<fp_t *, ExecSpace> weights,
    const bool forceEvaluation = false) {
  const fp_t one = 1.0;
  const fp_t minus_one = -1.0;
  const fp_t zero = 0.0;

  Kokkos::View<fp_t[spaceDimension], ExecSpace> xietaphi;
  for (int j = 0; j < spaceDimension; j++) {
    xietaphi(j) = 0.5;
  }
  // Something wrong here witht he compiler !
  // auto xcoor_sv = Kokkos::subview(
  //     Xcoor,
  //     Kokkos::make_pair(0, (int)FEspace::numberOfShapeFunctions),
  //     Kokkos::make_pair(0, (int)spaceDimension)
  // );
  Kokkos::View<fp_t[FEspace::numberOfShapeFunctions][spaceDimension],
               ExecSpace>
      xcoor_sv; // dim =
                // [FEspace::numberOfShapeFunctions][FEspace::dimensionality]
  for (int i = 0; i < FEspace::numberOfShapeFunctions; i++) {
    for (int j = 0; j < spaceDimension; j++) {
      xcoor_sv(i, j) = Xcoor(i, j);
    }
  }
  Kokkos::View<fp_t[FEspace::numberOfShapeFunctions], ExecSpace>
      sfv; // dim = [FEspace::numberOfShapeFunctions][1]
  Kokkos::View<fp_t[spaceDimension], ExecSpace> f; // dim = [1][spaceDimension]
  Kokkos::View<fp_t[FEspace::dimensionality][FEspace::numberOfShapeFunctions],
               ExecSpace>
      dN; // dim = [FEspace::dimensionality][FEspace::numberOfShapeFunctions]
  Kokkos::View<fp_t[FEspace::dimensionality][spaceDimension], ExecSpace>
      dNXcoor;
  Kokkos::View<fp_t[FEspace::dimensionality], ExecSpace>
      dfNum; // dim = [FEspace::dimensionality][1]
  Kokkos::View<fp_t[FEspace::dimensionality][FEspace::dimensionality],
               ExecSpace>
      h; // dim = [FEspace::dimensionality][FEspace::dimensionality]
  Kokkos::View<fp_t[FEspace::dimensionality], ExecSpace>
      dxietaphi; // dim = [FEspace::dimensionality][1]

  FEspace::UpdateShapeFunctionsValues(xietaphi(0), xietaphi(1), xietaphi(2),
                                      sfv);

  int cnt = 0;
  using namespace KokkosBatched;

  for (; cnt < MAX_NEWTON_ITER; ++cnt) {
    KokkosBlas::Experimental::serial_gemv('T', one, xcoor_sv, sfv, zero,
                                          f); // f= sfv.transpose() * xcoor;
    KokkosBlas::serial_axpy(minus_one, targetPoint,
                            f); // f -= targetPoint;

    FEspace::UpdateShapeFunctionsDerValues(xietaphi(0), xietaphi(1),
                                           xietaphi(2), dN);

    // dfNum = (f * ((dN * xcoor).transpose())).transpose();
    // here h = (dN * xcoor)
    SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
               Algo::Gemm::Unblocked>::invoke(one, dN, xcoor_sv, zero, dNXcoor);
    KokkosBlas::Experimental::serial_gemv('N', one, dNXcoor, f, zero, dfNum);

    // h  = (dN * xcoor) * (dN * xcoor).transpose();
    SerialGemm<Trans::NoTranspose, Trans::Transpose,
               Algo::Gemm::Unblocked>::invoke(one, dNXcoor, dNXcoor, zero, h);

    // if (x > 9) {
    //   h += DdfPart2<geoStruct>(-f, dN, xcoor, xietaphi);
    // }

    // ElemDimSolve(h, dfNum, dxietaphi) -> dxietaphi  = h^-1 * dfNum;
    // FiniteElement::Matrix<ExecSpace, FEspace::dimensionality>::InvInPlace(h);
    // KokkosBlas::Experimental::serial_gemv('N', one, h, dfNum, zero,
    // dxietaphi);
    SerialLU<Algo::LU::Unblocked>::invoke(h);
    SerialTrsv<Uplo::Lower, Trans::NoTranspose, Diag::Unit,
               Algo::Trsv::Unblocked>::invoke(one, h, dfNum);

    for (int j = 0; j < spaceDimension; j++) {
      xietaphi(j) -= dfNum(j);
    }

    FEspace::UpdateShapeFunctionsValues(xietaphi(0), xietaphi(1), xietaphi(2),
                                        sfv);

    fp_t norm2 = 0.0;
    for (int j = 0; j < FEspace::dimensionality; j++) {
      norm2 += dxietaphi(j) * dxietaphi(j);
    }
    if (norm2 < D_XI_ETA_PHI_TOL_SQUARED) {
      break;
    }
  }

  if (forceEvaluation) {
    for (int i = 0; i < FEspace::numberOfShapeFunctions; ++i) {
      weights(i) = sfv(i);
    }
    return true;
  }

  // if (x < MAX_NEWTON_ITER &&
  // Muscat::IsBaryCoordInsideGeo<geoStruct::geotype>(xietaphi)) {
  //     for (int i = 0; i < geoStruct::numberOfShapeFunctions; ++i) {
  //         weights(i) = n(i);
  //     }
  //     return true;
  return false;
}
} // namespace FiniteElements
} // namespace PACMAN
