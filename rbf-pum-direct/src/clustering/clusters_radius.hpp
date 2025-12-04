#pragma once

#include <Kokkos_Core.hpp>

#include "callbacks.hpp"
#include "interpolator.hxx"

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::find_radius(void)
{
    assert(this->_nodes_per_cluster > 0);
    const std::string _region_name = "RbfPumInterpolator::find_radius";
    Kokkos::Profiling::ScopedRegion region(_region_name);
    const ExecSpace execspace{};

    VectorView<Point> samples(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::samples"),
                              2 * Dim + 1);
    auto samples_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                Kokkos::HostSpace{}, samples);

    const Point lower = this->_source_bvh.bounds().minCorner();
    const Point upper = this->_source_bvh.bounds().maxCorner();
    Point center{};
    for (int axis = 0; axis < Dim; ++axis)
    {
        center[axis] = (lower[axis] + upper[axis]) / 2.;
    }

    samples_h(0) = Point{ center };
    for (int axis = 0; axis < Dim; ++axis)
    {
        Point sample = Point{ center };
        RbfPumFPType current_axis_length = std::fabs(upper[axis] - lower[axis]);
        sample[axis] -= current_axis_length * 0.25;
        samples_h(axis + 1) = Point{ sample };
        sample[axis] += current_axis_length * 0.5;
        samples_h(Dim + axis + 1) = Point{ sample };
    }

    Kokkos::deep_copy(samples, samples_h);

    Projection project_samples_to_input_predicate{ samples };
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
    VectorView<int> projected_samples_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::projected_samples_offsets"),
        0);

    this->_source_bvh.query(execspace, project_samples_to_input_predicate,
                            project_samples_to_input_callback,
                            projected_samples_values,
                            projected_samples_offsets);

    // Each sample matches exactly one projected point
    Kokkos::parallel_for(
        _region_name
            + "::p_for project samples on the source "
              "mesh",
        Kokkos::RangePolicy(execspace, 0, 2 * Dim + 1),
        KOKKOS_LAMBDA(const int& i) {
            samples(i) =
                projected_samples_values(projected_samples_offsets(i)).second;
        });
    Kokkos::fence();

    DistanceToKNearest vertices_per_sample_predicate{ this->_nodes_per_cluster,
                                                      samples };
    DistanceToKNearestCallback vertices_per_sample_callback{};

    VectorView<RbfPumFPType> squared_distances_values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::squared_distances_values"),
        0);
    VectorView<int> squared_distances_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::squared_distances_offsets"),
        0);
    ArborX::query(this->_source_bvh, execspace, vertices_per_sample_predicate,
                  vertices_per_sample_callback, squared_distances_values,
                  squared_distances_offsets);

    VectorView<RbfPumFPType> max_radii(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::max_radii"),
        2 * Dim + 1);
    Kokkos::parallel_for(
        _region_name + "::p_for sum max radius",
        Kokkos::RangePolicy(execspace, 0, 2 * Dim + 1),
        KOKKOS_LAMBDA(const int& i) {
            RbfPumFPType n_max = 0;
            for (int ii = squared_distances_offsets(i);
                 ii < squared_distances_offsets(i + 1); ++ii)
            {
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
    this->_radius = std::sqrt(max_radii_h(Dim));
}
