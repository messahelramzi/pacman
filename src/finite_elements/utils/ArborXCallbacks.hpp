//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX.hpp>
#include <algorithms/ArborX_ClosestPoint.hpp>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"
#include "finite_elements/utils/NewtonKokkos.hpp"

/**
 * @file ArborXCallbacks.hpp
 * @brief ArborX predicate/callback helpers for PACMAN finite-elements kernels.
 */

namespace PACMAN {
namespace FiniteElements {

/// @brief Wrapper for nearest-neighbor predicates built from 1D point views.
/// @tparam ViewType Kokkos view type containing point geometry values.
template <KokkosViewRank<1> ViewType> struct PointNearest { ViewType points; };

/// @brief Point-finite element intersection and interplolation callback.
/// @tparam ExecSpace Kokkos execution space.
/// @tparam Dim Spatial dimension (used with triangle point coordinates).
template <typename ExecSpace, int_t Dim>
struct PointFiniteElementInterpolation {
  using MemorySpace = typename ExecSpace::memory_space;

  Transfer<ExecSpace, Dim> transfer;
  Kokkos::View<int *, MemorySpace> parents;

  /// @brief ArborX callback: project one outside target point and emit output.
  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION auto operator()(const Predicate &predicate,
                                  const Value &value) const {

    const int predicate_index = ArborX::getData(predicate);
    const int_t curElem = parents(value.index); // RAMZI: is it needed ?
    const auto cellType = transfer.cellTypes(curElem);

    Kokkos::Array<fp_t, MaxNodesPerElt> warr;
    Kokkos::Array<coordinates_t, MaxNodesPerElt * Dim> Xarr;
    Kokkos::Array<coordinates_t, Dim> tpa;
    Kokkos::View<coordinates_t *, ExecSpace> tp(tpa.data(), Dim);

    for (int_t j = 0; j < Dim; j++) {
      tp(j) = transfer.targetPoints(predicate_index, j);
    }
    const auto localConn =
        Kokkos::subview(transfer.connValues,
                        Kokkos::make_pair(transfer.connOffsets(curElem),
                                          transfer.connOffsets(curElem + 1)));
    const offset_t nbConnNodes =
        transfer.connOffsets(curElem + 1) - transfer.connOffsets(curElem);
    Kokkos::View<fp_t *, ExecSpace> weights(warr.data(), nbConnNodes);
    Kokkos::View<coordinates_t **, ExecSpace> Xcoor(Xarr.data(), nbConnNodes,
                                                    Dim);
    for (int_t j = 0; j < nbConnNodes; ++j) {
      const auto nodeId = localConn(j);
      for (int_t k = 0; k < Dim; ++k) {
        Xcoor(j, k) = transfer.sourcePoints(nodeId)[k];
      }
    }
    if (ApplyNewtonOnElement<ExecSpace, Dim>(cellType, Xcoor, tp,
                                             weights /*, false*/)) {
      for (int_t j = 0; j < nbConnNodes; ++j) {
        const auto nodeId = localConn(j);
        transfer.targetValues(predicate_index) +=
            weights[j] * transfer.sourceValues(nodeId);
      }
      transfer.targetStatus(predicate_index) = TransferStatus::INTER;
      return ArborX::CallbackTreeTraversalControl::early_exit;
    } else {
      return ArborX::CallbackTreeTraversalControl::normal_continuation;
    }
  }
};

/// @brief Project outside target points onto nearest 3D skin triangle.
/// @tparam ExecSpace Kokkos execution space.
/// @tparam Dim Spatial dimension (used with triangle point coordinates).
template <typename ExecSpace, int_t Dim> struct PointTriangleProjection {
  using MemorySpace = typename ExecSpace::memory_space;

  Kokkos::View<coordinates_t **, MemorySpace> targetPointsPtr;
  Kokkos::View<int_t *, MemorySpace> parentElt;
  Kokkos::View<TransferStatus *, MemorySpace> status;
  Kokkos::View<int_t *, MemorySpace> skinParents;

  /// @brief ArborX callback: project one outside target point and emit output.
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value,
                                  const OutputFunctor &out) const {
    const int predicate_index = ArborX::getData(predicate);

    if (status(predicate_index) == TransferStatus::OUTSIDE) {
      ArborX::Point<Dim, coordinates_t> targetPoint;
      for (int_t k = 0; k < Dim; k++) {
        targetPoint[k] = targetPointsPtr(predicate_index, k);
      }

      auto triangle = value.value;

      using PointTag = ArborX::GeometryTraits::PointTag;
      using TriangleTag = ArborX::GeometryTraits::TriangleTag;
      using Point = decltype(targetPoint);
      using Triangle = decltype(triangle);

      ArborX::Details::Dispatch::distance<PointTag, TriangleTag, Point,
                                          Triangle>
          dist{};
      targetPoint = ArborX::Experimental::closestPoint(targetPoint, triangle);
      for (int_t k = 0; k < Dim; k++) {
        targetPointsPtr(predicate_index, k) = targetPoint[k];
      }
      parentElt(predicate_index) = skinParents(value.index);
      out(predicate_index);
    }
  }
};

/// @brief ArborX callback variant for extrapolation mode.
///
/// This callback records the parent element from the nearest skin primitive
/// without moving target point coordinates.
template <typename ExecSpace, int Dim> struct PointTriangleProjectionExtrapol {
  using MemorySpace = typename ExecSpace::memory_space;

  Kokkos::View<int_t *, MemorySpace> parentElt;
  Kokkos::View<TransferStatus *, MemorySpace> status;
  Kokkos::View<int_t *, MemorySpace> skinParents;

  /// @brief ArborX callback: assign nearest skin parent for outside points.
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value,
                                  const OutputFunctor &out) const {
    const int predicate_index = ArborX::getData(predicate);

    if (status(predicate_index) == TransferStatus::OUTSIDE) {
      parentElt(predicate_index) = skinParents(value.index);
      out(predicate_index);
    }
  }
};

/// @brief Generic callback extracting index from ArborX query values.
struct ExtractIndex {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, const Value &value,
                                  const OutputFunctor &out) const {
    out(value.index);
  }
};

/// @brief Predicate wrapper for nearest-neighbor queries over point clouds.
/// @tparam MemorySpace Kokkos memory space.
/// @tparam Dim Point-cloud dimensionality.
template <typename MemorySpace, int Dim> struct PointCloudNearest {
  Kokkos::View<coordinates_t **, MemorySpace> points;
};

/// @brief Callback assigning source value at nearest-neighbor index.
/// @tparam MemorySpace Kokkos memory space.
template <typename MemorySpace> struct NearestExtractIndex {
  Kokkos::View<fp_t *, MemorySpace> sourceValues;
  Kokkos::View<fp_t *, MemorySpace> targetValues;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value) const {
    const int predicate_index = ArborX::getData(predicate);
    targetValues(predicate_index) = sourceValues(value.index);
  }
};

/// @brief Predicate wrapper for point/box intersection queries.
/// @tparam MemorySpace Kokkos memory space.
/// @tparam Dim Point dimensionality.
template <typename MemorySpace, int_t Dim> struct PointIntersect {
  Kokkos::View<coordinates_t **, MemorySpace> points;
};

} // namespace FiniteElements

} // namespace PACMAN

namespace ArborX {
/// @brief ArborX access traits for nearest-neighbor predicates from 1D views.
template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::FiniteElements::PointNearest<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::FiniteElements::PointNearest<ViewType>;
  KOKKOS_FUNCTION
  static size_t size(const Self &self) { return self.points.extent(0); }
  KOKKOS_FUNCTION
  static auto get(const Self &self, size_t i) {
    return attach(nearest(self.points(i), 1), (int)i);
  }
};

/// @brief ArborX access traits for nearest-neighbor point-cloud predicates.
template <typename MemorySpace, int Dim>
struct AccessTraits<
    PACMAN::FiniteElements::PointCloudNearest<MemorySpace, Dim>> {
  using memory_space = MemorySpace;
  using Self = PACMAN::FiniteElements::PointCloudNearest<MemorySpace, Dim>;

  KOKKOS_FUNCTION
  static size_t size(const Self &self) { return self.points.extent(0); }
  KOKKOS_FUNCTION
  static auto get(const Self &self, size_t i) {
    Point<Dim, PACMAN::coordinates_t> point;
    for (int k = 0; k < Dim; k++) {
      point[k] = self.points(i, k);
    }
    return attach(nearest(point, 1), (int)i);
  }
};

/// @brief ArborX access traits for point-intersection predicates.
template <typename MemorySpace, int Dim>
struct AccessTraits<PACMAN::FiniteElements::PointIntersect<MemorySpace, Dim>> {
  using memory_space = MemorySpace;
  using Self = PACMAN::FiniteElements::PointIntersect<MemorySpace, Dim>;

  KOKKOS_FUNCTION
  static size_t size(const Self &self) { return self.points.extent(0); }
  KOKKOS_FUNCTION
  static auto get(const Self &self, size_t i) {
    Point<Dim, PACMAN::coordinates_t> point;
    for (int k = 0; k < Dim; k++) {
      point[k] = self.points(i, k);
    }
    return attach(intersects(point), (int)i);
  }
};

} // namespace ArborX
