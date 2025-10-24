#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <chrono>
#include <iostream>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include "interpolator.hpp"
#include "polynomial.hpp"
#include "rbf_functions.hpp"
#include "utils.hpp"

template <class scalar_type>
scalar_type franke_function(ArborX::Point<3, scalar_type> point)
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

template <int Dim, class Coordinates>
ArborX::Point<Dim, Coordinates>* get_points_from_vtu_grid(char* filename,
                                                          size_t* out_N)
{
    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(filename);
    reader->Update();
    vtkUnstructuredGrid* grid = reader->GetOutput();
    vtkPoints* points = grid->GetPoints();
    vtkIdType N = points->GetNumberOfPoints();
    ArborX::Point<Dim, Coordinates>* ret =
        (ArborX::Point<Dim, Coordinates>*)malloc(
            N * sizeof(ArborX::Point<Dim, Coordinates>));
    for (vtkIdType i = 0; i < N; ++i)
    {
        Coordinates coords[3];
        points->GetPoint(i, coords);
        auto p = ArborX::Point<Dim, Coordinates>{};
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
    using execution_space = Kokkos::DefaultExecutionSpace;
    using RbfFunctionBasisType = WendlandC2<scalar_type>;
    using PolynomialType = LinearPolynomial<execution_space, dim, scalar_type>;

    auto guard = Kokkos::ScopeGuard();
    {
        Kokkos::Profiling::ScopedRegion region("main::main");
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
        PolynomialType polynomial{};

        auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto interpolator =
            RbfPumInterpolator<execution_space, dim, scalar_type,
                               PolynomialType, RbfFunctionBasisType>(
                source, values, target, polynomial, rbf_function);
        auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();

        std::cout << "Source mesh: " << argv[1] << "(points: " << N << ")\n";
        std::cout << "Target mesh: " << argv[2] << "(points: " << M << ")\n";
        std::cout << "Time spent: " << (t2 - t1).count() / 1'000'000 << "ms"
                  << "\n";
        std::cout << interpolator.get_interpolator_details() << std::endl;

        free(source_grid);
        free(target_grid);
        free(values_host);
    }

    return 0;
}
