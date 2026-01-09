#pragma once

#include "callbacks.hpp"
#include "common/types.hpp"
#include "interpolator.hxx"
#include "utils/operators.hpp"

namespace PACMAN {
namespace RbfPum {
template <int_t X, int_t N> constexpr index_t CompileTimePow(void) {
  index_t prod = 1;
  for (index_t i = 0; i < N; ++i) {
    prod *= X;
  }
  return prod;
}

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

template <typename ValueType> struct RemoveTaggedCenters {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType &v) const { return v[0] != v[0]; }
};

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::CreateClusters(void) {
  assert(this->mRadius > 0);
  assert(this->mRelativeOverlap > 0 && this->mRelativeOverlap < 1);

  const std::string _region_name = "RbfPumInterpolator::create_clusters";
  Kokkos::Profiling::ScopedRegion region(_region_name);
  const ExecSpace execspace{};

  // 1. calcul de la BB sur les points source
  const Point lower = this->mSourceBvh.bounds().minCorner();
  const Point upper = this->mSourceBvh.bounds().maxCorner();

  // 2. calcul de la distance maximale entre 2 centres pour garantir l'overlap
  const fp_t spacing = std::sqrt(4.0 / Dim) * this->mRadius *
                       (fp_consts::one() - this->mRelativeOverlap);

  int_t shape[Dim];
  fp_t distances[Dim];
  fp_t min_distance = fp_consts::max();
  int_t nb_centers = 1;
  for (int_t axis = 0; axis < Dim; ++axis) {
    fp_t edge_length = upper[axis] - lower[axis];
    edge_length = (edge_length < 0) ? (-edge_length) : (edge_length);

    // 3-1. calcul du nombre de centres effectifs dans chaque direction
    shape[axis] = std::ceil(std::max(fp_consts::one(), edge_length / spacing));

    // 4. calcul des distances dx, dy, dz (en d dimensions)
    distances[axis] = edge_length / shape[axis];
    if (distances[axis] < min_distance) {
      min_distance = distances[axis];
    }

    // 3-2. On rajoute 1 centre dans chaque direction et on considère comme
    // premier centre le point `lower`
    nb_centers *= ++shape[axis];
  }

  // 5. création des centres candidats
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
        // a. calcul des coordonnées du point
        Kokkos::Array<int_t, Dim> coords{};
        for (int_t axis = Dim - 1; axis >= 0; --axis) {
          int_t prod = 1;
          for (int_t d = axis - 1; d >= 0; --d) {
            prod *= shape[d];
          }
          coords[axis] = i_copy / prod;
          i_copy -= coords[axis] * prod;
        }

        // b. calcul de l'indice zCurve
        int_t index = 0;
        int_t stride = 1;
        for (int_t k = Dim - 1; k >= 0; --k) {
          index += coords[k] * stride;
          stride *= shape[k];
        }

        // c. calcul de la position du point dans la BB
        centers_candidates(index) = Point{lower};
        for (int_t axis = 0; axis < Dim; ++axis) {
          centers_candidates(index)[axis] += distances[axis] * coords[axis];
        }
      });
  Kokkos::fence();

  // 6. on tag les clusters qui ne contiennent pas de point source donc ceux
  // pour lesquels: norm2(nearest(center), center) > radius
  TagEmptyCenters tag_empty_centers{centers_candidates};
  TagEmptyCentersCallback tag_empty_centers_callback{
      centers_candidates, this->mRadius * this->mRadius};
  this->mSourceBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 7. on tag les clusters qui ne contiennent pas de point target
  // donc ceux pour lesquels: norm2(nearest(center), center) > radius
  this->mTargetBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 8. on projette les centres non taggés sur leur point source respectif le
  // plus proche
  TransformToNearest transform_to_nearest{centers_candidates};
  TransformToNearestCallback transform_to_nearest_callback{centers_candidates};

  this->mSourceBvh.query(execspace, transform_to_nearest,
                         transform_to_nearest_callback);

  // 9. on retire les centres qui sont trop proches les uns des autres,
  // threshold = 0.4 * min(distances)
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

  // 10. on tag les clusters qui ne contiennent pas de point target
  // donc ceux pour lesquels: norm2(nearest(center), center) > radius
  tag_empty_centers = decltype(tag_empty_centers){centers_candidates};
  tag_empty_centers_callback = decltype(tag_empty_centers_callback){
      centers_candidates, this->mRadius * this->mRadius};
  this->mTargetBvh.query(execspace, tag_empty_centers,
                         tag_empty_centers_callback);

  // 11. on retire les centres taggés
  const auto end = Kokkos::Experimental::remove_if(
      _region_name + "::remove_if", execspace, centers_candidates,
      RemoveTaggedCenters<Point>{});
  const int_t dist = Kokkos::Experimental::distance(
      Kokkos::Experimental::begin(centers_candidates), end);
  Kokkos::resize(this->mClusters, dist);
}
} // namespace RbfPum

} // namespace PACMAN
