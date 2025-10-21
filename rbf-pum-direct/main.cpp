#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>

#include "interpolator.hpp"
#include "polynomial.hpp"
#include "rbf_functions.hpp"
#include "utils.hpp"

template <class scalar_type>
KOKKOS_INLINE_FUNCTION scalar_type
franke_function(ArborX::Point<3, scalar_type> point)
{
    scalar_type x = point[0];
    scalar_type y = point[1];
    scalar_type z = point[2];
    return 0.75
        * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0)
                     + (9.0 * y - 2.0) * (9.0 * y - 2.0)
                     + (9.0 * z - 2.0) * (9.0 * z - 2.0))
                   / 4.0)
        + 0.75
        * std::exp(-(((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                     + (9.0 * y + 1.0) / 10.0 + (9.0 * z + 1.0) / 10.0))
        + 0.5
        * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0)
                     + (9.0 * y - 3.0) * (9.0 * y - 3.0)
                     + ((9.0 * z - 5.0) * (9.0 * z - 5.0)) / 4.0))
        - 0.2
        * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)
                     + (9.0 * y - 7.0) * (9.0 * y - 7.0)
                     + (9.0 * z - 5.0) * (9.0 * z - 5.0)));
}

int main(void)
{
    auto guard = Kokkos::ScopeGuard();
    {
        using exec_space = Kokkos::DefaultExecutionSpace;
        using host_space = Kokkos::HostSpace;
        using scalar_type = double;
        const int dimensions = 3;
        using polynomial =
            LinearPolynomial<exec_space, dimensions, scalar_type>;
        using rbf_function = WendlandC2<scalar_type>;
        using point = ArborX::Point<dimensions, scalar_type>;
        exec_space execspace{};
        // 0.01 -> 0.001 (precice aste turbine)
        const size_t N = 34580;
        const size_t M = 338992;

        Kokkos::View<point*, exec_space> source(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "main::source"),
            N);
        Kokkos::View<scalar_type*, exec_space> values(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "main::values"),
            N);
        Kokkos::View<point*, exec_space> target(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "main::target"),
            M);

        Kokkos::Random_XorShift1024_Pool<> random_pool(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count());

        scalar_type lower = 0.;
        scalar_type upper = 1.;
        Kokkos::parallel_for(
            "fill example views - source & values",
            Kokkos::RangePolicy(execspace, 0, N),
            KOKKOS_LAMBDA(const size_t& i) {
                point p{};
                auto gen = random_pool.get_state();
                for (int d = 0; d < dimensions; ++d)
                {
                    p[d] = gen.drand(lower, upper);
                }
                random_pool.free_state(gen);
                source(i) = p;
                values(i) = franke_function(p);
            });
        Kokkos::parallel_for(
            "fill example views - target", Kokkos::RangePolicy(execspace, 0, M),
            KOKKOS_LAMBDA(const size_t& i) {
                point p{};
                auto gen = random_pool.get_state();
                for (int d = 0; d < dimensions; ++d)
                {
                    p[d] = gen.drand(lower, upper);
                }
                random_pool.free_state(gen);
                target(i) = p;
            });
        Kokkos::fence();

        auto f = rbf_function{};
        auto p = polynomial{};

        auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto r =
            RbfPumInterpolator<exec_space, dimensions, scalar_type, polynomial,
                               rbf_function>(source, values, target, p, f);
        auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();
        std::cout << "found _radius: " << r.get_radius() << std::endl;
        std::cout << "time: " << (t2 - t1).count() / 1000000 << "ms"
                  << std::endl;
    }

    return 0;
}
