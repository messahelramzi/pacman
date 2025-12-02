#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <chrono>
#include <iostream>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include "src/interpolator.hpp"

template <class ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
franke_function(ArborX::Point<3, ScalarType> point)
{
    ScalarType x = point[0];
    ScalarType y = point[1];
    ScalarType z = point[2];
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

template <int Dim, class ScalarType>
ArborX::Point<Dim, ScalarType>* get_points_from_vtu_grid(char* filename,
                                                         size_t* out_N)
{
    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(filename);
    reader->Update();
    vtkUnstructuredGrid* grid = reader->GetOutput();
    vtkPoints* points = grid->GetPoints();
    vtkIdType N = points->GetNumberOfPoints();
    ArborX::Point<Dim, ScalarType>* ret =
        (ArborX::Point<Dim, ScalarType>*)malloc(
            N * sizeof(ArborX::Point<Dim, ScalarType>));
    for (vtkIdType i = 0; i < N; ++i)
    {
        double coords[3];
        points->GetPoint(i, coords);
        auto p = ArborX::Point<Dim, ScalarType>{};
        for (int j = 0; j < Dim; ++j)
        {
            p[j] = coords[j];
        }
        ret[i] = p;
    }
    *out_N = N;
    return ret;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <source mesh.vtu> <target mesh.vtu>" << std::endl;
        return EXIT_FAILURE;
    }

    constexpr const int dim = 3;
    using scalar_type = double;
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using RbfFunctionBasisType = WendlandC6<scalar_type>;

    auto guard = Kokkos::ScopeGuard();
    {
        size_t N, M;
        auto source_grid =
            get_points_from_vtu_grid<dim, scalar_type>(argv[1], &N);
        auto values_host = (scalar_type*)malloc(N * sizeof(scalar_type));
        auto target_grid =
            get_points_from_vtu_grid<dim, scalar_type>(argv[2], &M);
        for (size_t i = 0; i < N; ++i)
        {
            values_host[i] = franke_function<scalar_type>(source_grid[i]);
        }

        Kokkos::View<ArborX::Point<dim, scalar_type>*, execution_space> source(
            Kokkos::view_alloc(execution_space{}, Kokkos::WithoutInitializing,
                               "main::source"),
            N);
        auto source_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                   Kokkos::HostSpace{}, source);
        Kokkos::View<scalar_type*, execution_space> values(
            Kokkos::view_alloc(execution_space{}, Kokkos::WithoutInitializing,
                               "main::values"),
            N);
        auto values_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                   Kokkos::HostSpace{}, values);

        Kokkos::View<ArborX::Point<dim, scalar_type>*, execution_space> target(
            Kokkos::view_alloc(execution_space{}, Kokkos::WithoutInitializing,
                               "main::target"),
            M);
        auto target_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                   Kokkos::HostSpace{}, target);

        for (size_t i = 0; i < N; ++i)
        {
            source_h(i) = source_grid[i];
            values_h(i) = values_host[i];
        }
        for (size_t i = 0; i < M; ++i)
        {
            target_h(i) = target_grid[i];
        }

        Kokkos::deep_copy(source, source_h);
        Kokkos::deep_copy(values, values_h);
        Kokkos::deep_copy(target, target_h);

        RbfFunctionBasisType rbf_function{};

        auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto interpolator =
            RbfPumInterpolator<execution_space, dim, scalar_type,
                               RbfFunctionBasisType>(source, values, target,
                                                     rbf_function);
        auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();

        Kokkos::View<scalar_type*, execution_space> reference(
            Kokkos::view_alloc(execution_space{}, Kokkos::WithoutInitializing,
                               "reference"),
            target.extent(0));

        std::cout << "Source mesh: " << argv[1] << "(points: " << N << ")\n";
        std::cout << "Target mesh: " << argv[2] << "(points: " << M << ")\n";
        std::cout << "Time spent: " << (t2 - t1).count() / 1'000'000.0 << "ms"
                  << "\n";
        std::cout << interpolator.get_interpolator_details() << std::endl;

        const auto interpolated_data = interpolator.out;

        Kokkos::parallel_for(
            "interpolate data", Kokkos::RangePolicy(execution_space{}, 0, M),
            KOKKOS_LAMBDA(const size_t& i) {
                reference(i) = franke_function(target(i));
            });
        Kokkos::fence();

        Kokkos::View<scalar_type*, execution_space> errors(
            Kokkos::view_alloc(execution_space{}, Kokkos::WithoutInitializing,
                               "main::errors"),
            reference.extent(0));

        Kokkos::parallel_for(
            "compute abs errors", Kokkos::RangePolicy(execution_space{}, 0, M),
            KOKKOS_LAMBDA(const size_t& i) {
                errors(i) = Kokkos::fabs(reference(i) - interpolated_data(i));
            });
        Kokkos::fence();

        Kokkos::sort(errors);
        scalar_type sum = 0.0;
        Kokkos::parallel_reduce(
            "sum", Kokkos::RangePolicy<execution_space>(0, M),
            KOKKOS_LAMBDA(const int i, scalar_type& update) {
                update += errors(i);
            },
            sum);
        scalar_type mean = sum / M;

        scalar_type min_val, max_val;
        Kokkos::deep_copy(min_val, Kokkos::subview(errors, 0));
        Kokkos::deep_copy(max_val,
                          Kokkos::subview(errors, errors.extent(0) - 1));

        auto worst_idx = [&](double pct) {
            std::size_t i = static_cast<std::size_t>(pct * M);
            if (i >= M)
                i = M - 1;
            return i;
        };

        std::size_t i1 = worst_idx(0.99);
        std::size_t i5 = worst_idx(0.95);
        std::size_t i10 = worst_idx(0.90);
        std::size_t i50 = worst_idx(0.50);

        scalar_type worst1, worst5, worst10, worst50;
        Kokkos::deep_copy(worst1, Kokkos::subview(errors, i1));
        Kokkos::deep_copy(worst5, Kokkos::subview(errors, i5));
        Kokkos::deep_copy(worst10, Kokkos::subview(errors, i10));
        Kokkos::deep_copy(worst50, Kokkos::subview(errors, i50));

        std::cout << std::setprecision(16);
        std::cout << "\n\n========== STATS ==========\n";
        std::cout << "avg: " << mean << "\n";
        std::cout << "min: " << min_val << "\n";
        std::cout << "max: " << max_val << "\n";
        std::cout << "med: " << worst50 << "\n";
        std::cout << "p90: " << worst10 << "\n";
        std::cout << "p95: " << worst5 << "\n";
        std::cout << "p99: " << worst1 << "\n";
        std::cout << "========== STATS ==========\n\n";

        free(source_grid);
        free(target_grid);
        free(values_host);
    }

    return 0;
}
