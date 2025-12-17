#pragma once

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <cctype>
#include <solver/rbf_functions.hpp>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#define ITERATIONRESULT_SUM_ID -1
#define ITERATIONRESULT_UNDEFINED_ID -2

struct IterationResult
{
    int id;
    std::string prefix;
    double avg;
    double min;
    double max;
    double med;
    double p90;
    double p95;
    double p99;
    double time_ms;

    IterationResult& operator+=(const IterationResult& rhs)
    {
        id = ITERATIONRESULT_SUM_ID;
        avg += rhs.avg;
        min += rhs.min;
        max += rhs.max;
        med += rhs.med;
        p90 += rhs.p90;
        p95 += rhs.p95;
        p99 += rhs.p99;
        time_ms += rhs.time_ms;
        return *this;
    }
};

IterationResult operator+(const IterationResult& lhs,
                          const IterationResult& rhs)
{
    IterationResult res{ ITERATIONRESULT_SUM_ID, lhs.prefix,
                         lhs.avg + rhs.avg,      lhs.min + rhs.min,
                         lhs.max + rhs.max,      lhs.med + rhs.med,
                         lhs.p90 + rhs.p90,      lhs.p95 + rhs.p95,
                         lhs.p99 + rhs.p99,      lhs.time_ms + rhs.time_ms };
    return res;
}

std::ostream& operator<<(std::ostream& os, const IterationResult& obj)
{
    if (obj.id >= 0)
    {
        os << obj.prefix << "id: " << obj.id << "\n";
    }
    os << obj.prefix << "avg: " << obj.avg << "\n";
    os << obj.prefix << "min: " << obj.min << "\n";
    os << obj.prefix << "max: " << obj.max << "\n";
    os << obj.prefix << "med: " << obj.med << "\n";
    os << obj.prefix << "p90: " << obj.p90 << "\n";
    os << obj.prefix << "p95: " << obj.p95 << "\n";
    os << obj.prefix << "p99: " << obj.p99 << "\n";
    os << obj.prefix << "time-ms: " << obj.time_ms;

    return os;
}

using AvailableExecSpaces = std::variant<
#if defined(KOKKOS_ENABLE_SERIAL)
    Kokkos::Serial
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    ,
    Kokkos::OpenMP
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    ,
    Kokkos::Threads
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    ,
    Kokkos::Cuda
#endif
#if defined(KOKKOS_ENABLE_HIP)
    ,
    Kokkos::HIP
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    ,
    Kokkos::SYCL
#endif
    >;

using AvailableFunctions =
    std::variant<WendlandC0<double>, WendlandC2<double>, WendlandC4<double>,
                 WendlandC6<double>, WendlandC8<double>>;

std::string str_tolower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

AvailableExecSpaces make_execspace(const std::string s)
{
    // clang-format off
    const std::string ts = str_tolower(s);
    if (ts == "serial")
    {
        #if defined(KOKKOS_ENABLE_SERIAL)
            return Kokkos::Serial{};
        #else
            std::cerr << "Kokkos::Serial is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SERIAL!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    if (ts == "openmp")
    {
        #if defined(KOKKOS_ENABLE_OPENMP)
            return Kokkos::OpenMP{};
        #else
            std::cerr << "Kokkos::OpenMP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_OPENMP!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    if (ts == "threads")
    {
        #if defined(KOKKOS_ENABLE_THREADS)
            return Kokkos::Threads{};
        #else
            std::cerr << "Kokkos::Threads is not enabled! Please recompile Kokkos with Kokkos_ENABLE_THREADS!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    if (ts == "cuda")
    {
        #if defined(KOKKOS_ENABLE_CUDA)
            return Kokkos::Cuda{};
        #else
            std::cerr << "Kokkos::Cuda is not enabled! Please recompile Kokkos with Kokkos_ENABLE_CUDA!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    if (ts == "hip")
    {
        #if defined(KOKKOS_ENABLE_HIP)
            return Kokkos::HIP{};
        #else
            std::cerr << "Kokkos::HIP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_HIP!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    if (ts == "sycl")
    {
        #if defined(KOKKOS_ENABLE_SYCL)
            return Kokkos::SYCL{};
        #else
            std::cerr << "Kokkos::SYCL is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SYCL!" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    std::cerr << "execspace: Serial, OpenMP, Threads, Cuda, Hip, Sycl"
              << std::endl;
    exit(EXIT_FAILURE);
    // clang-format on
}

template <typename fp_t>
AvailableFunctions make_rbffunction(const std::string s)
{
    std::string ts = str_tolower(s);
    if (ts == "wendlandc0")
    {
        return WendlandC0<fp_t>{};
    }
    if (ts == "wendlandc2")
    {
        return WendlandC2<fp_t>{};
    }
    if (ts == "wendlandc4")
    {
        return WendlandC4<fp_t>{};
    }
    if (ts == "wendlandc6")
    {
        return WendlandC6<fp_t>{};
    }
    if (ts == "wendlandc8")
    {
        return WendlandC8<fp_t>{};
    }
    std::cerr << "function: WendlandC0, WendlandC2, WendlandC4, "
                 "WendlandC6, WendlandC8"
              << std::endl;
    exit(EXIT_FAILURE);
}

template <int Dim, typename RbfPumFPType>
ArborX::Point<Dim, RbfPumFPType>* get_points_from_vtu_grid(const char* filename,
                                                           size_t* out_N)
{
    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(filename);
    reader->Update();
    vtkUnstructuredGrid* grid = reader->GetOutput();
    vtkPoints* points = grid->GetPoints();
    vtkIdType N = points->GetNumberOfPoints();
    ArborX::Point<Dim, RbfPumFPType>* ret =
        (ArborX::Point<Dim, RbfPumFPType>*)malloc(
            N * sizeof(ArborX::Point<Dim, RbfPumFPType>));
    for (vtkIdType i = 0; i < N; ++i)
    {
        double coords[3];
        points->GetPoint(i, coords);
        auto p = ArborX::Point<Dim, RbfPumFPType>{};
        for (int j = 0; j < Dim; ++j)
        {
            p[j] = coords[j];
        }
        ret[i] = p;
    }
    *out_N = N;
    return ret;
}

template <typename RbfPumFPType>
KOKKOS_INLINE_FUNCTION RbfPumFPType
franke_function(ArborX::Point<3, RbfPumFPType>& point)
{
    const RbfPumFPType x = point[0];
    const RbfPumFPType y = point[1];
    const RbfPumFPType z = point[2];
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
                     + (9.0 * z - 5.0) * (9.0 * z - 5.0))
                   / 4.0)
        - 0.2
        * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)
                     + (9.0 * y - 7.0) * (9.0 * y - 7.0)
                     + (9.0 * z - 5.0) * (9.0 * z - 5.0)));
}

template <typename RbfPumFPType>
KOKKOS_INLINE_FUNCTION RbfPumFPType
franke_function(ArborX::Point<2, RbfPumFPType>& point)
{
    const RbfPumFPType x = point[0];
    const RbfPumFPType y = point[1];
    return 0.75
        * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0)
                     + (9.0 * y - 2.0) * (9.0 * y - 2.0))
                   / 4.0)
        + 0.75
        * std::exp(-(((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                     + (9.0 * y + 1.0) / 10.0))
        + 0.5
        * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0)
                     + (9.0 * y - 3.0) * (9.0 * y - 3.0))
                   / 4.0)
        - 0.2
        * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)
                     + (9.0 * y - 7.0) * (9.0 * y - 7.0)));
}

template <typename RbfPumFPType>
KOKKOS_INLINE_FUNCTION RbfPumFPType
franke_function(ArborX::Point<1, RbfPumFPType>& point)
{
    const RbfPumFPType x = point[0];
    return 0.75 * std::exp(-((9.0 * x - 2.0) * (9.0 * x - 2.0)) / 4.0)
        + 0.75 * std::exp(-((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0)
        + 0.5 * std::exp(-((9.0 * x - 7.0) * (9.0 * x - 7.0)) / 4.0)
        - 0.2 * std::exp(-((9.0 * x - 4.0) * (9.0 * x - 4.0)));
}
