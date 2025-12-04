#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T rbfpum_fma(T a, T x, T b)
{
#if defined(KOKKOS_ENABLE_CUDA)
    return Kokkos::fma(a, x, b);
#elif defined(FP_FAST_FMA)
    return std::fma(a, x, b);
#else
    return a * x + b;
#endif
}

template <typename RbfPumFPType>
__forceinline__ auto auto_fp_format(void)
{
    return std::setprecision(std::numeric_limits<RbfPumFPType>::max_digits10);
}

template <typename ExecSpace>
KOKKOS_FORCEINLINE_FUNCTION constexpr bool is_host_accessible(void)
{
    return Kokkos::SpaceAccessibility<
        Kokkos::DefaultHostExecutionSpace,
        typename ExecSpace::memory_space>::accessible;
}

template <typename T>
using base_type =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;

void print_cuda_memory_usage()
{
    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    std::cout << "used: " << (cuda_total - cuda_free) / 1'000'000.0 << "/"
              << (cuda_total) / 1'000'000.0 << "MB" << std::endl;
}

template <typename ViewType>
void print_size_of_view(ViewType& v)
{
    size_t size = 0;
    if (v.rank() == 1)
    {
        size = ViewType::required_allocation_size(v.extent(0));
    }
    else if (v.rank() == 2)
    {
        size = ViewType::required_allocation_size(v.extent(0), v.extent(1));
    }
    else if (v.rank() == 3)
    {
        size = ViewType::required_allocation_size(v.extent(0), v.extent(1),
                                                  v.extent(2));
    }
    else
    {
        size = ViewType::required_allocation_size(v.size());
    }
    std::cout << v.label() << " size: " << size << "B = " << size / 1'000'000.0
              << "MB" << std::endl;
}

template <typename ViewType>
void print_view(ViewType& v, std::string sep = " ")
{
    using data_type = typename ViewType::non_const_value_type;
    std::cout << auto_fp_format<data_type>();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    for (size_t i = 0; i < m.extent(0); ++i)
    {
        std::cout << m(i) << sep;
    }
    std::cout << std::endl;
    std::cout << v.label() << ".extent(0): " << m.extent(0) << std::endl;
}

template <typename ViewType, typename OffsView>
void export_coeffs(const ViewType& coeffs, const OffsView& offs,
                   const std::string& folder)
{
    using data_type = typename ViewType::non_const_value_type;
    std::cout << auto_fp_format<data_type>();
    auto coeffs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coeffs);
    auto offs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offs);
    for (int k = 0; k < offs_host.extent(0) - 1; ++k)
    {
        const auto access_index = offs_host(k);
        const auto next_index = offs_host(k + 1);
        const auto n = next_index - access_index;
        const std::string name =
            folder + "coeffs_" + std::to_string(k) + ".csv";
        std::fstream file{ name,
                           file.in | file.out | file.trunc | file.binary };
        if (!file.is_open())
        {
            std::cerr << name << " failed to open!" << std::endl;
            std::cerr << strerror(errno) << "\n";
            continue;
        }
        file << std::fixed << auto_fp_format<data_type>();
        for (int i = 0; i < n - 1; ++i)
        {
            file << coeffs_host(access_index + i) << ",";
        }
        file << coeffs_host(access_index + n - 1) << "\n";
    }
}

template <typename ViewType>
void export_matrix(const ViewType& matrix, const int& n,
                   const std::string& name)
{
    using data_type = typename ViewType::non_const_value_type;
    std::cout << auto_fp_format<data_type>();
    std::fstream file{ name, file.in | file.out | file.trunc | file.binary };
    if (!file.is_open())
    {
        std::cerr << name << " failed to open!" << std::endl;
        std::cerr << strerror(errno) << "\n";
        return;
    }
    file << std::fixed << auto_fp_format<data_type>();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n - 1; ++j)
        {
            const auto val = matrix(i * n + j);
            sum += val * val;
            file << val << ",";
        }
        const auto val = matrix(i * n + (n - 1));
        sum += val * val;
        file << val << "\n";
    }
    file.close();
    std::cout << "Wrote a matrix to " << name << std::endl;
    std::cout << "Matrix norm: " << std::sqrt(sum) << std::endl;
}

template <typename ViewType, typename OffsType>
void export_systems(const ViewType& data, const OffsType& offs,
                    const std::string& folder = "./mats")
{
    const auto data_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);
    const auto offs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offs);
    const size_t K = offs_host.extent(0) - 1;
    for (size_t k = 0; k < K; ++k)
    {
        const auto range = Kokkos::make_pair(offs_host(k), offs_host(k + 1));
        const auto A = Kokkos::subview(data_host, range);
        const std::filesystem::path name =
            folder + "/" + "mat_" + std::to_string(k) + ".csv";
        export_matrix(A, (int)(std::sqrt(offs_host(k + 1) - offs_host(k))),
                      name.string());
    }
}
