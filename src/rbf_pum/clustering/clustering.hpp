//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

#include "callbacks.hpp"
#include "common/types.hpp"
#include "interpolator.hxx"
#include "utils/operators.hpp"

namespace PACMAN {
namespace RbfPum {

/// @brief A pow function computed at compile time.
/// @tparam X The value X of $X^N$
/// @tparam N The value N of $X^N$
/// @return The value $X^N$ at compile time
template <int_t X, int_t N> constexpr index_t CompileTimePow(void) {
  index_t prod = 1;
  for (index_t i = 0; i < N; ++i) {
    prod *= X;
  }
  return prod;
}

/// @brief Return the indices offsets of the spatial neighbors given the shape
/// of the space.
/// @tparam Dim The dimension of the space.
/// @param shape The number of regions in each direction (each axis).
/// @return An std::array of size 3^{Dim} - 1 which contains the offset to apply
/// to each point to get its neighbors.
template <int_t Dim> constexpr auto ZCurveNeighborOffsets(int_t shape[Dim]) {
  constexpr index_t N = CompileTimePow<3, Dim>() - 1;
  std::array<int_t, N> result{};

  std::array<int_t, Dim> offset{};
  index_t out = 0;

  const index_t total = N + 1;

  for (index_t i = 0; i < total; ++i) {
    index_t t = i;
    bool all_zero = true;

    for (int_t d = 0; d < Dim; ++d) {
      int_t digit = static_cast<int_t>(t % 3);
      t /= 3;
      digit -= 1;
      offset[d] = digit;
      if (digit != 0) {
        all_zero = false;
      }
    }

    if (!all_zero) {
      // Indice zCurve du offset calculé
      int_t index = 0;
      index_t stride = 1;
      for (int_t k = Dim - 1; k >= 0; --k) {
        index += offset[k] * stride;
        stride *= shape[k];
      }
      result[out++] = index;
    }
  }
  return result;
}

/// @brief: 2D specialization of `ZCurveNeighborOffsets` for faster computations
template <> constexpr auto ZCurveNeighborOffsets<2>(int_t shape[2]) {
  std::array<int_t, 8> result{};

  const std::array<std::array<int_t, 2>, 8> offsets = {
      {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};

  index_t out = 0;
  for (const auto offset : offsets) {
    int_t index = 0;
    index_t stride = 1;
    for (int_t k = 1; k >= 0; --k) {
      index += offset[k] * stride;
      stride *= shape[k];
    }
    result[out++] = index;
  }

  return result;
}

/// @brief: 3D specialization of `ZCurveNeighborOffsets` for faster computations
template <> constexpr auto ZCurveNeighborOffsets<3>(int_t shape[3]) {
  std::array<int_t, 26> result{};

  const std::array<std::array<int_t, 3>, 26> offsets = {
      {{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, 0},  {-1, 0, 1},
       {-1, 0, -1},  {-1, 1, 0},  {-1, 1, 1},  {-1, 1, -1}, {0, -1, -1},
       {0, -1, 0},   {0, -1, 1},  {0, 0, 1},   {0, 0, -1},  {0, 1, 0},
       {0, 1, 1},    {0, 1, -1},  {1, -1, -1}, {1, -1, 0},  {1, -1, 1},
       {1, 0, 0},    {1, 0, 1},   {1, 0, -1},  {1, 1, 0},   {1, 1, 1},
       {1, 1, -1}}};

  index_t out = 0;
  for (const auto offset : offsets) {
    int_t index = 0;
    index_t stride = 1;
    for (int_t k = 2; k >= 0; --k) {
      index += offset[k] * stride;
      stride *= shape[k];
    }
    result[out++] = index;
  }

  return result;
}

/// @brief A functor which returns `true` is a given point is tagged for removal
/// (=> if its first dimension coordinate is NaN)
template <typename ValueType> struct RemoveTaggedCenters {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType &v) const { return v[0] != v[0]; }
};

/// @brief Creates the clusters centers view using the same heuristics as
/// [preCICE](github.com/precice/precice) does.
FULL_TEMPLATE
void TEMPLATED_CLASSNAME::CreateClusters(void) {
  assert(this->mRadius > 0);
  assert(this->mRelativeOverlap > 0 && this->mRelativeOverlap < 1);

  const std::string _region_name = "RbfPumInterpolator::create_clusters";
  Kokkos::Profiling::ScopedRegion region(_region_name);
  const ExecSpace execspace{};

  // 1. computation of the source points bounding box
  const Point lower = this->mSourceBvh.bounds().minCorner();
  const Point upper = this->mSourceBvh.bounds().maxCorner();

  // 2. compatation of the max distance between two points to guarantee the
  // given overlap ratio
  const fp_t spacing = std::sqrt(4.0 / Dim) * this->mRadius *
                       (fp_consts::one() - this->mRelativeOverlap);

  int_t shape[Dim];
  fp_t distances[Dim];
  fp_t min_distance = fp_consts::max();
  int_t nb_centers = 1;
  for (int_t axis = 0; axis < Dim; ++axis) {
    fp_t edge_length = upper[axis] - lower[axis];
    edge_length = (edge_length < 0) ? (-edge_length) : (edge_length);

    // 3-1. computation of the number of centers we can fit in each direction
    shape[axis] = std::ceil(std::max(fp_consts::one(), edge_length / spacing));

    // 4. computation of the distances dx, dy, dz, the space between centers in
    // each direction
    distances[axis] = edge_length / shape[axis];
    if (distances[axis] < min_distance) {
      min_distance = distances[axis];
    }

    // we begin fitting centers at the beginning of the bounding box
    nb_centers *= ++shape[axis];
  }

  // 5. we create a view with all of the clusters centers we want to try
  this->mClusters = decltype(this->mClusters)(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         "this->mClusters"),
      nb_centers);
  auto centers_candidates = this->mClusters;

  Kokkos::parallel_for(
      _region_name + "::p_for fill centers candidates",
      Kokkos::RangePolicy(execspace, 0, nb_centers),
      KOKKOS_LAMBDA(const index_t &i) {
        index_t i_copy = i;
        // a. we compute the coordinates of the point on the centers grid
        Kokkos::Array<int_t, Dim> coords{};
        for (int_t axis = Dim - 1; axis >= 0; --axis) {
          int_t prod = 1;
          for (int_t d = axis - 1; d >= 0; --d) {
            prod *= shape[d];
          }
          coords[axis] = i_copy / prod;
          i_copy -= coords[axis] * prod;
        }

        // b. computation of the zCurve index
        int_t index = 0;
        int_t stride = 1;
        for (int_t k = Dim - 1; k >= 0; --k) {
          index += coords[k] * stride;
          stride *= shape[k];
        }

        // c. computation of the spatial coordinates of the point in space
        centers_candidates(index) = Point{lower};
        for (int_t axis = 0; axis < Dim; ++axis) {
          centers_candidates(index)[axis] += distances[axis] * coords[axis];
        }
      });
  Kokkos::fence();

  // 6. we tag for removal clusters which do not contain source points
  // = all clusters with norm2(nearest(center), center) > radius
  TagEmptyCenters tag_empty_centers{centers_candidates};
  TagEmptyCentersCallback tag_empty_centers_callback{
      centers_candidates, this->mRadius * this->mRadius};
  this->mSourceBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 7. same step but checking for target points in clusters
  this->mTargetBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 8. we transform each center candidate to its nearest point on the source
  // mesh/source points cloud
  TransformToNearest transform_to_nearest{centers_candidates};
  TransformToNearestCallback transform_to_nearest_callback{centers_candidates};

  this->mSourceBvh.query(execspace, transform_to_nearest,
                         transform_to_nearest_callback);

  // 9. we remove centers that are too close from each others (can be improved
  // and taken to parallel computations on GPU) threshold = 0.4 * min(distances)
  fp_t threshold = 0.4 * min_distance;
  threshold *= threshold;
  auto centers_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          centers_candidates);
  auto neighbors_offsets = ZCurveNeighborOffsets<Dim>(shape);
  for (index_t i = 0; i < centers_host.extent(0); ++i) {
    // is not NaN = is not tagged
    if (centers_host(i)[0] == centers_host(i)[0]) {
      for (auto off : neighbors_offsets) {
        if (!off) {
          continue;
        }
        int_t neighbor_id = i + off;
        if (neighbor_id >= 0 && neighbor_id < centers_host.extent_int(0)) {
          auto center = centers_host(i);
          auto neighbor = centers_host(neighbor_id);
          // is not NaN = is not tagged AND is too close
          if (neighbor[0] == neighbor[0] &&
              SquaredDifference(center, neighbor) < threshold) {
            centers_host(i)[0] = NAN;
            break;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(centers_candidates, centers_host);

  // 10. we tag for removal the clusters which do not contain any target point:
  // = all clusters with norm2(nearest(center), center) > radius
  tag_empty_centers = decltype(tag_empty_centers){centers_candidates};
  tag_empty_centers_callback = decltype(tag_empty_centers_callback){
      centers_candidates, this->mRadius * this->mRadius};
  this->mTargetBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 11. we remove tagged centers
  const auto end = Kokkos::Experimental::remove_if(
      _region_name + "::remove_if", execspace, centers_candidates,
      RemoveTaggedCenters<Point>{});
  const int_t dist = Kokkos::Experimental::distance(
      Kokkos::Experimental::begin(centers_candidates), end);
  Kokkos::resize(this->mClusters, dist);
}
} // namespace RbfPum

} // namespace PACMAN
