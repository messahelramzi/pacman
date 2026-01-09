#pragma once

#include <ArborX.hpp>

#include "common/concepts.hpp"
#include "common/transfer.hxx"
#include "common/types.hpp"

namespace PACMAN {
namespace FiniteElements {
template <KokkosViewRank<1> ViewType> struct PointNearest {
  ViewType points;
};

template <typename ExecSpace, int_t spaceDimension>
struct PointTriangleProjection {
  using MemorySpace = typename ExecSpace::memory_space;

  Kokkos::View<coordinates_t **, MemorySpace> targetPointsPtr;
  Kokkos::View<int_t *, MemorySpace> parentElt;
  Kokkos::View<TransferStatus *, MemorySpace> status;
  Kokkos::View<int_t *, MemorySpace> skinParents;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value,
                                  const OutputFunctor &out) const {
    const int predicate_index = ArborX::getData(predicate);

    if (status(predicate_index) == TransferStatus::UNDEFINED) {
      ArborX::Point<spaceDimension, coordinates_t> targetPoint;
      for (int_t k = 0; k < spaceDimension; k++) {
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
      targetPoint =
          dist.closest_point(targetPoint, triangle.a, triangle.b, triangle.c);
      for (int_t k = 0; k < spaceDimension; k++) {
        targetPointsPtr(predicate_index, k) = targetPoint[k];
      }
      parentElt(predicate_index) = skinParents(value.index);
      out(predicate_index);
    }
  }
};

template <typename ExecSpace, int spaceDimension>
struct PointTriangleProjectionExtrapol {
  using MemorySpace = typename ExecSpace::memory_space;

  Kokkos::View<int_t *, MemorySpace> parentElt;
  Kokkos::View<TransferStatus *, MemorySpace> status;
  Kokkos::View<int_t *, MemorySpace> skinParents;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(const Predicate &predicate,
                                  const Value &value,
                                  const OutputFunctor &out) const {
    const int predicate_index = ArborX::getData(predicate);

    if (status(predicate_index) == TransferStatus::UNDEFINED) {
      parentElt(predicate_index) = skinParents(value.index);
      out(predicate_index);
    }
  }
};

struct ExtractIndex {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, const Value &value,
                                  const OutputFunctor &out) const {
    out(value.index);
  }
};

template <typename MemorySpace, int spaceDimension> struct PointCloudNearest {
  Kokkos::View<coordinates_t **, MemorySpace> points;
};

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

template <typename MemorySpace, int_t spaceDimension> struct PointIntersect {
  Kokkos::View<coordinates_t **, MemorySpace> points;
};

} // namespace FiniteElements

} // namespace PACMAN

namespace ArborX {
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

template <typename MemorySpace, int spaceDimension>
struct AccessTraits<
    PACMAN::FiniteElements::PointCloudNearest<MemorySpace, spaceDimension>> {
  using memory_space = MemorySpace;
  using Self =
      PACMAN::FiniteElements::PointCloudNearest<MemorySpace, spaceDimension>;

  KOKKOS_FUNCTION
  static size_t size(const Self &self) { return self.points.extent(0); }
  KOKKOS_FUNCTION
  static auto get(const Self &self, size_t i) {
    Point<spaceDimension, PACMAN::coordinates_t> point;
    for (int k = 0; k < spaceDimension; k++) {
      point[k] = self.points(i, k);
    }
    return attach(nearest(point, 1), (int)i);
  }
};

template <typename MemorySpace, int spaceDimension>
struct AccessTraits<
    PACMAN::FiniteElements::PointIntersect<MemorySpace, spaceDimension>> {
  using memory_space = MemorySpace;
  using Self =
      PACMAN::FiniteElements::PointIntersect<MemorySpace, spaceDimension>;

  KOKKOS_FUNCTION
  static size_t size(const Self &self) { return self.points.extent(0); }
  KOKKOS_FUNCTION
  static auto get(const Self &self, size_t i) {
    Point<spaceDimension, PACMAN::coordinates_t> point;
    for (int k = 0; k < spaceDimension; k++) {
      point[k] = self.points(i, k);
    }
    return attach(intersects(point), (int)i);
  }
};
} // namespace ArborX
