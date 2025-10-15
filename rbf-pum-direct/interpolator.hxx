#ifndef INTERPOLATOR_HXX
#define INTERPOLATOR_HXX

#include <ArborX_Box.hpp>
#include <ArborX_Sphere.hpp>
#include <Kokkos_Core.hpp>

template <typename ExecSpace, int Dim = 3, class Coordinates = double>
class RbfPumInterpolator
{
    using Point = ArborX::Point<Dim, Coordinates>;
    using Box = ArborX::Box<Dim, Coordinates>;
    using PointsView = Kokkos::View<Point*, ExecSpace>;
    using ClustersView = Kokkos::View<Point**, ExecSpace>;

public:
    RbfPumInterpolator(PointsView source, PointsView target);
    void create_clusters(void);
    void find_radius(void);
    Coordinates get_radius() const;
    ClustersView _clusters;

private:
    double _radius;
    const int _nodes_per_cluster = 50;
    const double _relative_overlap = 0.15;
    const size_t _clustering_rd_samples = 100;
    Box _bd;
    PointsView _source;
    PointsView _target;
    Point no_data{};
};

#endif /* ! INTERPOLATOR_HXX */
