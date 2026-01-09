#pragma once

#include <ArborX.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "utils/operators.hpp"

namespace PACMAN {
namespace RbfPum {
template <KokkosViewRank<1> ViewType> struct DistanceToKNearest {
  const int_t k;
  ViewType samples;
};

struct DistanceToKNearestCallback {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate predicate, Value const &value,
                                  OutputFunctor const &out) const {
    out(SquaredDifference(ArborX::getData(predicate), value.value));
  }
};

template <KokkosViewRank<1> ViewType> struct Projection {
  ViewType centers;
};

struct ProjectionCallback {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate predicate, Value const &value,
                                  OutputFunctor const &out) const {
    // <centre, projection>
    out(Kokkos::make_pair(ArborX::getData(predicate), value.value));
  }
};

template <KokkosViewRank<1> ViewType> struct TagEmptyCenters {
  ViewType centersCandidates;
};

template <KokkosViewRank<1> ViewType> struct TagEmptyCentersCallback {
  ViewType centersCandidates;
  fp_t threshold;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate predicate,
                                  Value const &value) const {
    const int center_index = ArborX::getData(predicate);
    // The center should be removed
    if (SquaredDifference(centersCandidates(center_index), value.value) >
        threshold) {
      // Only one match per center, so centersCandidates(center_index) is
      // accessed only once
      // No need of atomic operation here
      centersCandidates(center_index)[0] = NAN;
    }
  }
};

template <KokkosViewRank<1> ViewType> struct TransformToNearest {
  ViewType centersCandidates;
};

template <KokkosViewRank<1> ViewType> struct TransformToNearestCallback {
  ViewType centersCandidates;
  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate predicate,
                                  Value const &value) const {
    // The center must be projected to the nearest point (value)
    centersCandidates(ArborX::getData(predicate)) = value.value;
  }
};

template <KokkosViewRank<1> ViewType> struct GetClustersPoints {
  ViewType centers;
  coordinates_t radius;
};

struct GetClustersPointsCallback {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate /* predicate */, Value const &value,
                                  OutputFunctor const &out) const {
    out(value.index);
  }
};

} // namespace RbfPum
} // namespace PACMAN

namespace ArborX {
template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::RbfPum::DistanceToKNearest<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::RbfPum::DistanceToKNearest<ViewType>;
  static KOKKOS_FUNCTION size_t size(const Self &self) {
    return self.samples.extent(0);
  }
  static KOKKOS_FUNCTION auto get(const Self &self, size_t i) {
    return attach(nearest(self.samples(i), self.k), self.samples(i));
  }
};

template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::RbfPum::Projection<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::RbfPum::Projection<ViewType>;
  static KOKKOS_FUNCTION size_t size(const Self &self) {
    return self.centers.extent(0);
  }
  static KOKKOS_FUNCTION auto get(const Self &self, size_t i) {
    return attach(nearest(self.centers(i), 1), self.centers(i));
  }
};

template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::RbfPum::TagEmptyCenters<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::RbfPum::TagEmptyCenters<ViewType>;
  using Point = typename ViewType::non_const_value_type;
  static KOKKOS_FUNCTION size_t size(const Self &self) {
    return self.centersCandidates.extent(0);
  }
  static KOKKOS_FUNCTION auto get(const Self &self, size_t i) {
    // isNaN
    if (self.centersCandidates(i)[0] != self.centersCandidates(i)[0]) {
      return attach(nearest(Point{}, 0), -1);
    }
    return attach(nearest(self.centersCandidates(i), 1), (int)i);
  }
};

template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::RbfPum::TransformToNearest<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::RbfPum::TransformToNearest<ViewType>;
  using Point = typename ViewType::non_const_value_type;
  static KOKKOS_FUNCTION size_t size(const Self &self) {
    return self.centersCandidates.extent(0);
  }
  static KOKKOS_FUNCTION auto get(const Self &self, size_t i) {
    // isNaN
    if (self.centersCandidates(i)[0] != self.centersCandidates(i)[0]) {
      return attach(nearest(Point{}, 0), -1);
    }
    return attach(nearest(self.centersCandidates(i), 1), (int)i);
  }
};

template <PACMAN::KokkosViewRank<1> ViewType>
struct AccessTraits<PACMAN::RbfPum::GetClustersPoints<ViewType>> {
  using memory_space = typename ViewType::memory_space;
  using Self = PACMAN::RbfPum::GetClustersPoints<ViewType>;
  static KOKKOS_FUNCTION size_t size(const Self &self) {
    return self.centers.extent(0);
  }
  static KOKKOS_FUNCTION auto get(const Self &self, size_t i) {
    return intersects(Sphere(self.centers(i), self.radius));
  }
};
} // namespace ArborX
