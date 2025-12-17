#pragma once

#include <Kokkos_Core.hpp>
#include <cctype>
#include <solver/rbf_functions.hpp>

#define ADD_EXECSPACES_ENTRY(E) execspaces.attr(#E) = ExecSpaces::E
#define ADD_RBF_FUNCTIONS_ENTRY(F) rbf_functions.attr(#F) = RbfFunctions::F

struct ExecSpaces
{
    static constexpr const char SERIAL = 0;
    static constexpr const char OPENMP = 1;
    static constexpr const char THREADS = 2;
    static constexpr const char CUDA = 3;
    static constexpr const char HIP = 4;
    static constexpr const char SYCL = 5;
};

struct RbfFunctions
{
    static constexpr const char WENDLANDC0 = 0;
    static constexpr const char WENDLANDC2 = 1;
    static constexpr const char WENDLANDC4 = 2;
    static constexpr const char WENDLANDC6 = 3;
    static constexpr const char WENDLANDC8 = 4;
};

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

AvailableExecSpaces make_execspace(const char s)
{
    // clang-format off
    switch (s) {
        case ExecSpaces::SERIAL:
            #if defined(KOKKOS_ENABLE_SERIAL)
                return Kokkos::Serial{};
            #else
                std::cerr << "Kokkos::Serial is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SERIAL!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        case ExecSpaces::OPENMP:
            #if defined(KOKKOS_ENABLE_OPENMP)
                return Kokkos::OpenMP{};
            #else
                std::cerr << "Kokkos::OpenMP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_OPENMP!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        case ExecSpaces::THREADS:
            #if defined(KOKKOS_ENABLE_THREADS)
                return Kokkos::Threads{};
            #else
                std::cerr << "Kokkos::Threads is not enabled! Please recompile Kokkos with Kokkos_ENABLE_THREADS!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        case ExecSpaces::CUDA:
            #if defined(KOKKOS_ENABLE_CUDA)
                return Kokkos::Cuda{};
            #else
                std::cerr << "Kokkos::Cuda is not enabled! Please recompile Kokkos with Kokkos_ENABLE_CUDA!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        case ExecSpaces::HIP:
            #if defined(KOKKOS_ENABLE_HIP)
                return Kokkos::HIP{};
            #else
                std::cerr << "Kokkos::HIP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_HIP!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        case ExecSpaces::SYCL:
            #if defined(KOKKOS_ENABLE_SYCL)
                return Kokkos::SYCL{};
            #else
                std::cerr << "Kokkos::SYCL is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SYCL!" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        default:
            std::cerr << "execspace: Serial, OpenMP, Threads, Cuda, HIP, SYCL"
                    << std::endl;
            exit(EXIT_FAILURE);
    }
    // clang-format on
}

template <typename fp_t>
AvailableFunctions make_rbffunction(const char s)
{
    switch (s)
    {
    case RbfFunctions::WENDLANDC0:
        return WendlandC0<fp_t>{};
    case RbfFunctions::WENDLANDC2:
        return WendlandC2<fp_t>{};
    case RbfFunctions::WENDLANDC4:
        return WendlandC4<fp_t>{};
    case RbfFunctions::WENDLANDC6:
        return WendlandC6<fp_t>{};
    case RbfFunctions::WENDLANDC8:
        return WendlandC8<fp_t>{};
    default:
        std::cerr << "function: WendlandC0, WendlandC2, WendlandC4, "
                     "WendlandC6, WendlandC8"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}