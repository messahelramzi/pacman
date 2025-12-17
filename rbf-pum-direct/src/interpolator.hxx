#pragma once

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>

#include "concepts.hpp"
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
    using MemSpace = typename ExecSpace::memory_space;

    template <typename inner_type>
    using VectorView = Kokkos::View<inner_type*, MemSpace>;

    template <typename inner_type>
    using MatrixView = Kokkos::View<inner_type**, MemSpace>;

public:
    // Constructor
    template <KokkosViewRank<1> SourceViewType,
              KokkosViewRank<1> ValuesViewType,
              KokkosViewRank<1> TargetViewType>
    RbfPumInterpolator(SourceViewType& source, ValuesViewType& values,
                       TargetViewType& target,
                       RbfFunctionBasisType& rbf_function,
                       int nodes_per_cluster = 50,
                       double relative_overlap = 0.15,
                       double rbf_support_radius = 0.1);

    // Internal routines
    void find_radius(void);
    void create_clusters(void);
    void solve_systems(void);

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
    int _nodes_per_cluster;
    double _relative_overlap;
    double _support_radius;

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
