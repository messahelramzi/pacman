#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>

#include "interpolator.hpp"
#include "polynomial.hpp"
#include "rbf_functions.hpp"
#include "utils.hpp"

int main(void)
{
    auto guard = Kokkos::ScopeGuard();
    {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using HostSpace = Kokkos::HostSpace;
        using fp_type = double;
        ExecSpace execspace{};
        // 0.01 -> 0.001 (precice aste turbine)
        // const size_t N = 3458;
        // const size_t M = 338992;
        const size_t N = 3458;
        const size_t M = 33899;
        Kokkos::View<ArborX::Point<3, fp_type>*, ExecSpace> source(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "source points"),
            N);
        Kokkos::View<ArborX::Point<3, fp_type>*, ExecSpace> target(
            Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                               "target points"),
            M);
        Kokkos::parallel_for(
            "fill source and target",
            Kokkos::RangePolicy<ExecSpace>(execspace, 0, N + M),
            KOKKOS_LAMBDA(const int& i) {
                if (i < N)
                {
                    source(i) = ArborX::Point<3, fp_type>(
                        { (fp_type)i, (fp_type)i, (fp_type)i });
                }
                if (i < M)
                {
                    target(i) = ArborX::Point<3, fp_type>(
                        { -(fp_type)i, -(fp_type)i, -(fp_type)i });
                }
            });
        Kokkos::fence();

        auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
        WendlandC2<fp_type> w2;
        LinearPolynomial<ExecSpace, 3, fp_type> poly;
        RbfPumInterpolator r =
            RbfPumInterpolator<ExecSpace, 3, fp_type, decltype(poly),
                               decltype(w2)>(source, target, poly, w2);
        auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();

        std::cout << "t2 - t1: " << (t2.count() - t1.count()) / 1000000.0
                  << "ms" << std::endl;
        std::cout << "found radius: " << r.get_radius() << std::endl;
    }

    return 0;
}
