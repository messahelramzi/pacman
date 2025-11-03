#ifndef INTERPOLATOR_HXX
#define INTERPOLATOR_HXX

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>

#include "polynomial.hpp"
#include "rbf_functions.hpp"

#define FULL_TEMPLATE                                                          \
    template <typename ExecSpace, int Dim, class Coordinates,                  \
              typename PolynomialType, typename RbfFunctionBasisType>

#define TEMPLATED_CLASSNAME                                                    \
    RbfPumInterpolator<ExecSpace, Dim, Coordinates, PolynomialType,            \
                       RbfFunctionBasisType>

FULL_TEMPLATE
class RbfPumInterpolator
{
    using Point = ArborX::Point<Dim, Coordinates>;
    using Box = ArborX::Box<Dim, Coordinates>;
    using PointsView = Kokkos::View<Point*, ExecSpace>;
    using ClustersView = Kokkos::View<Point**, ExecSpace>;
    using Polynomial = Polynomial<ExecSpace, Dim, Coordinates>;
    using RbfFunctionBasis = RbfFunctionBasis<Coordinates>;

public:
    RbfPumInterpolator(PointsView source,
                       Kokkos::View<Coordinates*, ExecSpace> values,
                       PointsView target, PolynomialType polynomial,
                       RbfFunctionBasisType rbf_function);
    void find_radius(void);
    void create_clusters(void);
    void prepare_interpolation(void);
    Coordinates interpolate_at(const Point& target) const;
    void interpolate(PointsView& target) const;
    std::string get_interpolator_details(void) const;

private:
    double _radius;
    const int _nodes_per_cluster = 50;
    const double _relative_overlap = 0.15;
    const bool _project_to_input = true;
    PointsView _source;
    Kokkos::View<Coordinates*, ExecSpace> _values;
    PointsView _target;
    ArborX::BoundingVolumeHierarchy<typename PointsView::memory_space, Point>
        _source_bvh;
    ArborX::BoundingVolumeHierarchy<typename PointsView::memory_space, Point>
        _target_bvh;
    Point no_data;
    ClustersView _clusters;
    Kokkos::View<size_t*, ExecSpace> _nb_values_per_cluster;
    PolynomialType _polynomial;
    RbfFunctionBasisType _rbf_function;
    WendlandC2<Coordinates> _weighting_function;
    Kokkos::View<Coordinates**, ExecSpace> _coeffs;
};

#endif /* ! INTERPOLATOR_HXX */
