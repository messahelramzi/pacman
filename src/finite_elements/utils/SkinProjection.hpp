//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
//

#pragma once

#include <ArborX_Point.hpp>
#include <ArborX_Triangle.hpp>
#include <Kokkos_Core.hpp>
#include <limits>

#include "common/types.hpp"
#include "finite_elements/utils/ArborXCallbacks.hpp"

namespace PACMAN {
namespace FiniteElements {

KOKKOS_INLINE_FUNCTION ::ArborX::Point<2, fp_t>
closest_point_to_edge(const ::ArborX::Point<2, fp_t> &p,
                      const ::ArborX::Point<2, fp_t> &a,
                      const ::ArborX::Point<2, fp_t> &b) {
  const auto ab = b - a;
  const auto ap = p - a;
  fp_t t = (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]);
  auto tclamp = Kokkos::max(0.0, Kokkos::min(1.0, t));
  ::ArborX::Point<2, fp_t> ret = {a[0] + ab[0] * tclamp, a[1] + ab[1] * tclamp};
  return ret;
};

template <typename ExecSpace>
void ComputeProjectionOn3DSkin(
    Transfer<ExecSpace, 3> &transfer,
    Kokkos::View<int_t **, typename ExecSpace::memory_space> col2d_d,
    Kokkos::View<fp_t **, typename ExecSpace::memory_space> val2d_d,
    Kokkos::View<int_t *, typename ExecSpace::memory_space> countEntries,
    bool extrapol = false) {
  using MemorySpace = typename ExecSpace::memory_space;
  using Triangle = ::ArborX::Triangle<3, fp_t>;
  using Point = ::ArborX::Point<3, fp_t>;

  ExecSpace execSpace{};

  auto nodesPtr = transfer.nodes;
  auto targetPointsPtr = transfer.targetPoints;
  auto targetElemPtr = transfer.targetElement;
  auto targetStatusPtr = transfer.targetStatus;
  auto elementsTypePtr = transfer.cellTypes;
  auto connValPtr = transfer.connectivity_values;
  auto connOffPtr = transfer.connectivity_offsets;
  auto skinFacesPtr = transfer.skinFaces;
  auto skinParentsPtr = transfer.skinParents;

  Kokkos::Profiling::pushRegion("Build triangle BoundingVolumeHierarchy");
  auto nbTri = transfer.skinFaces.extent(0);
  Kokkos::View<Triangle *, MemorySpace> skinFacesView(
      Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing,
                         "Skin ArborX Triangles"),
      nbTri);
  Kokkos::parallel_for(
      "Fill ArborX triangles",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, nbTri),
      KOKKOS_LAMBDA(int i) {
        for (int d = 0; d < 3; d++) {
          skinFacesView(i).a[d] = nodesPtr(skinFacesPtr(i, 0))[d];
          skinFacesView(i).b[d] = nodesPtr(skinFacesPtr(i, 1))[d];
          skinFacesView(i).c[d] = nodesPtr(skinFacesPtr(i, 2))[d];
        }
      });
  ArborX::BoundingVolumeHierarchy skinBVH(
      execSpace, ArborX::Experimental::attach_indices(skinFacesView));
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::View<int_t *, MemorySpace> values("values", 0);
  Kokkos::View<int_t *, MemorySpace> offsets("offsets", 0);

  Kokkos::Profiling::pushRegion("Point projection on triangle");
  PointCloudNearest<MemorySpace, 3> pcn{targetPointsPtr};
  if (extrapol) {
    skinBVH.query(execSpace, pcn,
                  PointTriangleProjectionExtrapol<ExecSpace, 3>{
                      targetElemPtr, targetStatusPtr, skinParentsPtr},
                  values, offsets);
  } else {
    skinBVH.query(
        execSpace, pcn,
        PointTriangleProjection<ExecSpace, 3>{targetPointsPtr, targetElemPtr,
                                              targetStatusPtr, skinParentsPtr},
        values, offsets);
  }
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  auto statusOutside =
      (extrapol) ? TransferStatus::EXTRAP : TransferStatus::CLAMP;

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Compute projection values ",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, values.extent(0)),
      KOKKOS_LAMBDA(int i) {
        auto predicate_index = values(i);
        Eigen::Matrix<fp_t, 1, 3> tp;
        for (int j = 0; j < 3; j++) {
          tp(0, j) = targetPointsPtr(predicate_index, j);
        }
        // primitive intersected bounding box finite element data
        const int_t curElem = targetElemPtr(predicate_index);
        const auto elemType = elementsTypePtr(curElem);
        Eigen::Matrix<fp_t, maxNodePerElt, 3> Xcoor;
        Xcoor.setZero();
        auto localConn = Kokkos::subview(
            connValPtr,
            Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
        int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
        for (int j = 0; j < nbConnNodes; ++j) {
          auto nodeId = localConn(j);
          for (int k = 0; k < 3; ++k) {
            Xcoor(j, k) = nodesPtr(nodeId)[k];
          }
        }
        auto val2d_sv = Kokkos::subview(val2d_d, predicate_index,
                                        Kokkos::make_pair(0, nbConnNodes));
        auto isInside = ApplyNewtonOnElement<MemorySpace, 3>(
            CellTypes(i) /*elemType, Xcoor, tp, val2d_sv, true*/);
        for (int j = 0; j < nbConnNodes; ++j) {
          col2d_d(predicate_index, j) = localConn(j);
        }
        countEntries(predicate_index) = nbConnNodes;
        targetStatusPtr(predicate_index) = statusOutside;
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return;
};

template <typename ExecSpace>
void ComputeProjectionOn2DSkin(
    Transfer<ExecSpace, 2> &transfer,
    Kokkos::View<int_t **, typename ExecSpace::memory_space> col2d_d,
    Kokkos::View<fp_t **, typename ExecSpace::memory_space> val2d_d,
    Kokkos::View<int_t *, typename ExecSpace::memory_space> countEntries,
    bool extrapol = false) {
  Kokkos::Profiling::pushRegion("Compute query of nearest skin");

  using MemorySpace = typename ExecSpace::memory_space;
  using Point = ArborX::Point<2, fp_t>;

  ExecSpace execSpace{};

  auto skinFacesPtr = transfer.skinFaces;
  auto skinParentsPtr = transfer.skinParents;
  auto nodesPtr = transfer.nodes;
  auto targetPointsPtr = transfer.targetPoints;
  auto targetElemPtr = transfer.targetElement;
  auto targetStatusPtr = transfer.targetStatus;
  auto CellTypesPtr = transfer.CellTypes;
  auto connValPtr = transfer.connectivity_values;
  auto connOffPtr = transfer.connectivity_offsets;

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Retrieve closest bar",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, targetPointsPtr.extent(0)),
      KOKKOS_LAMBDA(int i) {
        if (targetStatusPtr(i) == TransferStatus::UNDEFINED) {
          const Point target = {targetPointsPtr(i, 0), targetPointsPtr(i, 1)};
          fp_t mindistsqr = fp_consts::max();
          int_t closestId = -1;
          Point closest;
          for (int s = 0; s < skinFacesPtr.extent(0); s++) {
            auto p1 = nodesPtr(skinFacesPtr(s, 0));
            auto p2 = nodesPtr(skinFacesPtr(s, 1));
            auto cur = closest_point_to_edge(target, p1, p2);
            fp_t dx = cur[0] - target[0];
            fp_t dy = cur[1] - target[1];
            fp_t curdistsqr = dx * dx + dy * dy;
            if (curdistsqr < mindistsqr) {
              closest = cur;
              mindistsqr = curdistsqr;
              closestId = s;
            }
          }
          if (!extrapol) {
            targetPointsPtr(i, 0) = closest[0];
            targetPointsPtr(i, 1) = closest[1];
          }
          targetElemPtr(i) = skinParentsPtr(closestId);
          const int_t curElem = targetElemPtr(i);
          const auto elemType = elementsTypePtr(curElem);
          Eigen::Matrix<fp_t, maxNodePerElt, 2> Xcoor;
          Xcoor.setZero();
          auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
          for (int j = 0; j < nbConnNodes; ++j) {
            auto nodeId = localConn(j);
            for (int k = 0; k < 2; ++k) {
              Xcoor(j, k) = nodesPtr(nodeId)[k];
            }
          }
          Eigen::Matrix<fp_t, 1, 2> tp;
          for (int j = 0; j < 2; j++) {
            tp(0, j) = targetPointsPtr(i, j);
          }
          auto val2d_sv =
              Kokkos::subview(val2d_d, i, Kokkos::make_pair(0, nbConnNodes));
          auto isInside = ApplyNewtonOnElement<MemorySpace, 2>(
              CellTypes(i) /*elemType, Xcoor, tp, val2d_sv, true*/);
          for (int j = 0; j < nbConnNodes; ++j) {
            col2d_d(i, j) = localConn(j);
          }
          countEntries(i) = nbConnNodes;
          targetStatusPtr(i) = TransferStatus::CLAMP;
        }
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  return;
};

template <typename ExecSpace>
void ComputeProjectionOn1DSkin(
    Transfer<ExecSpace, 1> &transfer,
    Kokkos::View<int_t **, typename ExecSpace::memory_space> col2d_d,
    Kokkos::View<fp_t **, typename ExecSpace::memory_space> val2d_d,
    Kokkos::View<int_t *, typename ExecSpace::memory_space> countEntries,
    bool extrapol = false) {
  Kokkos::Profiling::pushRegion("Compute query of nearest skin");

  using MemorySpace = typename ExecSpace::memory_space;
  using Point = ArborX::Point<1, fp_t>;

  ExecSpace execSpace{};

  auto skinFacesPtr = transfer.skinFaces;
  auto nodesPtr = transfer.nodes;
  auto targetPointsPtr = transfer.targetPoints;
  auto targetElemPtr = transfer.targetElement;
  auto targetStatusPtr = transfer.targetStatus;
  auto skinParentsPtr = transfer.skinParents;
  auto elementsTypePtr = transfer.CellTypes;
  auto connValPtr = transfer.connectivity_values;
  auto connOffPtr = transfer.connectivity_offsets;

  Kokkos::Profiling::pushRegion(
      "Compute projection and interpolation coefficient");
  Kokkos::parallel_for(
      "Retrieve closest point",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, targetPointsPtr.extent(0)),
      KOKKOS_LAMBDA(int i) {
        if (targetStatusPtr(i) == TransferStatus::UNDEFINED) {
          const Point target = {targetPointsPtr(i, 0)};
          fp_t mindist = Kokkos::Experimental::finite_max<fp_t>::value;
          int_t closestId = -1;
          Point closest;
          for (int s = 0; s < skinFacesPtr.extent(0); s++) {
            auto p = nodesPtr(skinFacesPtr(s, 0));
            auto curdist = Kokkos::abs(p[0] - target[0]);
            if (curdist < mindist) {
              closest = p;
              mindist = curdist;
              closestId = s;
            }
          }
          if (!extrapol) {
            targetPointsPtr(i, 0) = closest[0];
          }
          targetElemPtr(i) = skinParentsPtr(closestId);
          const int_t curElem = targetElemPtr(i);
          const auto elemType = elementsTypePtr(curElem);

          Eigen::Matrix<fp_t, maxNodePerElt, 1> Xcoor;
          Xcoor.setZero();
          auto localConn = Kokkos::subview(
              connValPtr,
              Kokkos::make_pair(connOffPtr(curElem), connOffPtr(curElem + 1)));
          int nbConnNodes = connOffPtr(curElem + 1) - connOffPtr(curElem);
          for (int j = 0; j < nbConnNodes; ++j) {
            auto nodeId = localConn(j);
            for (int k = 0; k < 1; ++k) {
              Xcoor(j, k) = nodesPtr(nodeId)[k];
            }
          }

          Eigen::Matrix<fp_t, 1, 1> tp;
          for (int j = 0; j < 1; j++) {
            tp(0, j) = targetPointsPtr(i, j);
          }
          auto val2d_sv =
              Kokkos::subview(val2d_d, i, Kokkos::make_pair(0, nbConnNodes));
          auto isInside = ApplyNewtonOnElement<MemorySpace, 1>(
              CellTypes(i) /*elemType, Xcoor, tp, val2d_sv, true*/);
          for (int j = 0; j < nbConnNodes; ++j) {
            col2d_d(i, j) = localConn(j);
          }
          countEntries(i) = nbConnNodes;
          targetStatusPtr(i) = TransferStatus::CLAMP;
        }
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  return;
};

} // namespace FiniteElements
} // namespace PACMAN
