//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include <ArborX.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "utils/operators.hpp"

namespace PACMAN {
namespace RbfPum {

/// @brief Support struct for an ArborX predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct DistanceToKNearest {
  const int_t k;
  ViewType samples;
};

/// @brief Custom ArborX callback which returns the squared distance between a
/// cluster center and its nearest points
struct DistanceToKNearestCallback {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate predicate, Value const &value,
                                  OutputFunctor const &out) const {
    out(SquaredDifference(ArborX::getData(predicate), value.value));
  }
};

/// @brief Support struct for an ArborX predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct Projection { ViewType centers; };

/// @brief A custom ArborX callback which returns the nearest mesh point of a
/// given value (the nearest projection of it)
struct ProjectionCallback {
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate predicate, Value const &value,
                                  OutputFunctor const &out) const {
    // <center, projection>
    out(Kokkos::make_pair(ArborX::getData(predicate), value.value));
  }
};

/// @brief Support struct for an ArborX predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct TagEmptyCenters {
  ViewType centersCandidates;
};

/// @brief A custon ArborX callback which tags empty regions centers, by setting
/// the `x` axis value of the center to `NAN`
/// @tparam ViewType A `Kokkos::View` of rank 1
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

/// @brief Support struct for an ArborX predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct TransformToNearest {
  ViewType centersCandidates;
};

/// @brief A custom ArborX callback which performs a nearest point projection
/// for all the `ArborX::Point` in the given view
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct TransformToNearestCallback {
  ViewType centersCandidates;
  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate predicate,
                                  Value const &value) const {
    // The center must be projected to the nearest point (value)
    centersCandidates(ArborX::getData(predicate)) = value.value;
  }
};

/// @brief Support struct for an ArborX predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
template <KokkosViewRank<1> ViewType> struct GetClustersPoints {
  ViewType centers;
  coordinates_t radius;
};

/// @brief A custon ArborX callback which returns the indices of the points
/// inside of each region (according to the position of source/target points in
/// the transfer views)
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

/// @brief An `ArborX::AccessTraits` specialization used for the
/// `DistanceToKNearest` custom predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
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

/// @brief An `ArborX::AccessTraits` specialization used for the `Projection`
/// custom predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
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

/// @brief An `ArborX::AccessTraits` specialization used for the
/// `TagEmptyCenters` custom predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
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

/// @brief An `ArborX::AccessTraits` specialization used for the
/// `TransformToNearest` custom predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
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

/// @brief An `ArborX::AccessTraits` specialization used for the
/// `GetClustersPoints` custom predicate
/// @tparam ViewType A `Kokkos::View` of rank 1
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
