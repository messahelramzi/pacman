#pragma once

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>

#include "solver/rbf_functions.hpp"

#define FULL_TEMPLATE                                                          \
    template <typename ExecSpace, int Dim, typename RbfPumFPType,              \
              typename RbfFunctionBasisType>

#define TEMPLATED_CLASSNAME                                                    \
    RbfPumInterpolator<ExecSpace, Dim, RbfPumFPType, RbfFunctionBasisType>

FULL_TEMPLATE
class RbfPumInterpolator
{
    using Point = ArborX::Point<Dim, RbfPumFPType>;

    template <typename inner_type>
    using VectorView = Kokkos::View<inner_type*, ExecSpace>;

    template <typename inner_type>
    using MatrixView = Kokkos::View<inner_type**, ExecSpace>;

    template <typename inner_type>
    using TensorView = Kokkos::View<inner_type***, ExecSpace>;

public:
    // Constructor
    RbfPumInterpolator(VectorView<Point>& source,
                       VectorView<RbfPumFPType>& values,
                       VectorView<Point>& target,
                       RbfFunctionBasisType& rbf_function);

    // Internal routines
    void find_radius(void);
    void create_clusters(void);
    constexpr void solve_systems(void);
    void device_solve_systems(void);
    void host_solve_systems(void);

    // Debug routines
    std::string get_interpolator_details(void) const;

    // Getters
    KOKKOS_INLINE_FUNCTION auto get_source_i(const size_t& i) const
    {
        return this->_source(i);
    }
    KOKKOS_INLINE_FUNCTION auto get_target_i(const size_t& i) const
    {
        return this->_target(i);
    }
    KOKKOS_INLINE_FUNCTION auto get_values_i(const size_t& i) const
    {
        return this->_values(i);
    }
    KOKKOS_INLINE_FUNCTION auto get_centers_i(const size_t& i) const
    {
        return this->_clusters(i);
    }

    VectorView<RbfPumFPType> out;

private:
    double _radius;
    const int _nodes_per_cluster = 50;
    const double _relative_overlap = 0.15;
    const double _support_radius = 0.1;

    ArborX::BoundingVolumeHierarchy<typename VectorView<Point>::memory_space,
                                    ArborX::PairValueIndex<Point, unsigned int>>
        _source_bvh;
    ArborX::BoundingVolumeHierarchy<typename VectorView<Point>::memory_space,
                                    ArborX::PairValueIndex<Point, unsigned int>>
        _target_bvh;
    RbfFunctionBasisType _rbf_function;
    VectorView<Point> _clusters;
    VectorView<Point> _source;
    VectorView<Point> _target;
    VectorView<RbfPumFPType> _coeffs;
    VectorView<RbfPumFPType> _values;
    WendlandC2<RbfPumFPType> _weighting_function;
};
