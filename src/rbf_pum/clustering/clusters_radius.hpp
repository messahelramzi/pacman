#pragma once

#include <Kokkos_Core.hpp>

#include "callbacks.hpp"
#include "common/types.hpp"
#include "interpolator.hxx"

namespace PACMAN {
namespace RbfPum {
FULL_TEMPLATE
void TEMPLATED_CLASSNAME::FindRadius(void) {
  assert(this->mNodesPerCluster > 0);
  const std::string _region_name = "RbfPumInterpolator::find_radius";
  Kokkos::Profiling::ScopedRegion region(_region_name);
  const ExecSpace execspace{};

  VectorView<Point> samples(Kokkos::view_alloc(execspace,
                                               Kokkos::WithoutInitializing,
                                               _region_name + "::samples"),
                            2 * Dim + 1);
  auto samples_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                 Kokkos::HostSpace{}, samples);

  const Point lower = this->mSourceBvh.bounds().minCorner();
  const Point upper = this->mSourceBvh.bounds().maxCorner();
  Point center{};
  for (int_t axis = 0; axis < Dim; ++axis) {
    center[axis] = (lower[axis] + upper[axis]) / 2.;
  }

  samples_host(0) = Point{center};
  for (int_t axis = 0; axis < Dim; ++axis) {
    Point sample = Point{center};
    coordinates_t current_axis_length = std::abs(upper[axis] - lower[axis]);
    sample[axis] -= current_axis_length * 0.25;
    samples_host(axis + 1) = Point{sample};
    sample[axis] += current_axis_length * 0.5;
    samples_host(Dim + axis + 1) = Point{sample};
  }

  Kokkos::deep_copy(samples, samples_host);

  Projection project_samples_to_input_predicate{samples};
  ProjectionCallback project_samples_to_input_callback{};

  /* The returned values are pairs containing the point and its projection on
   *  the source mesh.
   *  Kokkos::pair<point, projection>
   */
  using ProjectionRet = Kokkos::pair<Point, Point>;
  VectorView<ProjectionRet> projected_samples_values(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         _region_name + "::projected_samples_values"),
      0);
  VectorView<offset_t> projected_samples_offsets(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         _region_name + "::projected_samples_offsets"),
      0);

  this->mSourceBvh.query(execspace, project_samples_to_input_predicate,
                         project_samples_to_input_callback,
                         projected_samples_values, projected_samples_offsets);

  // Each sample matches exactly one projected point
  Kokkos::parallel_for(
      _region_name + "::p_for project samples on the source "
                     "mesh",
      Kokkos::RangePolicy(execspace, 0, 2 * Dim + 1),
      KOKKOS_LAMBDA(const offset_t &i) {
        samples(i) =
            projected_samples_values(projected_samples_offsets(i)).second;
      });
  Kokkos::fence();

  DistanceToKNearest vertices_per_sample_predicate{this->mNodesPerCluster,
                                                   samples};
  DistanceToKNearestCallback vertices_per_sample_callback{};

  VectorView<fp_t> squared_distances_values(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         _region_name + "::squared_distances_values"),
      0);
  VectorView<offset_t> squared_distances_offsets(
      Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                         _region_name + "::squared_distances_offsets"),
      0);
  this->mSourceBvh.query(execspace, vertices_per_sample_predicate,
                         vertices_per_sample_callback, squared_distances_values,
                         squared_distances_offsets);

  VectorView<fp_t> max_radii(Kokkos::view_alloc(execspace,
                                                Kokkos::WithoutInitializing,
                                                _region_name + "::max_radii"),
                             2 * Dim + 1);
  Kokkos::parallel_for(
      _region_name + "::p_for sum max radius",
      Kokkos::RangePolicy(execspace, 0, 2 * Dim + 1),
      KOKKOS_LAMBDA(const int_t &i) {
        fp_t n_max = fp_consts::zero();
        for (offset_t ii = squared_distances_offsets(i);
             ii < squared_distances_offsets(i + 1); ++ii) {
          n_max = (n_max > squared_distances_values(ii))
                      ? n_max
                      : squared_distances_values(ii);
        }
        max_radii(i) = n_max;
      });
  Kokkos::fence();
  Kokkos::sort(max_radii);
  auto max_radii_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, max_radii);
  this->mRadius = std::sqrt(max_radii_h(Dim));
}
} // namespace RbfPum

} // namespace PACMAN
