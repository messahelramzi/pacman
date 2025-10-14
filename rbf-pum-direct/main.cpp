#include <chrono>
#include <iostream>

#include "Kokkos_Core.hpp"
#include "interpolator.hpp"
#include "utils.hpp"
int main(void)
{
    auto guard = Kokkos::ScopeGuard();
    {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using HostSpace = Kokkos::HostSpace;
        using fp_type = double;
        ExecSpace execspace{};
        const int N = 2500;
        const int M = 2500;
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
            Kokkos::RangePolicy<ExecSpace>(execspace, 0, N),
            KOKKOS_LAMBDA(const int i) {
                source(i) = ArborX::Point<3, fp_type>(
                    { (fp_type)i, (fp_type)i, (fp_type)i });

                target(i) = ArborX::Point<3, fp_type>(
                    { -(fp_type)i, -(fp_type)i, -(fp_type)i });
            });
        Kokkos::fence();

        auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto r = RbfPumInterpolator<ExecSpace, 3, fp_type>(source, target);
        auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();

        HostSpace hostspace{};
        auto m = Kokkos::create_mirror_view_and_copy(hostspace, r._clusters);
        for (size_t i = 0; i < m.extent(0); ++i) {
            auto cluster = m(i);
            std::cout << cluster << std::endl;
        }

        std::cout << "t2 - t1: " << (t2.count() - t1.count()) / 1000000.0
                  << "ms" << std::endl;
        std::cout << "found radius: " << r.get_radius() << std::endl;
    }

    return 0;
}
