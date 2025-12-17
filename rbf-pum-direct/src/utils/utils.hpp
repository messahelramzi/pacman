#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#if defined(Kokkos_ENABLE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#endif

#define SEND_TO_HOST(view)                                                     \
    auto h_##view = Kokkos::create_mirror_view_and_copy(                       \
        Kokkos::DefaultHostExecutionSpace{}, view)

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T rbfpum_fma(const T a, const T x, const T b)
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
inline auto auto_fp_format(void)
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
#if defined(Kokkos_ENABLE_CUDA)
    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    std::cout << "used: " << (cuda_total - cuda_free) / 1'000'000.0 << "/"
              << (cuda_total) / 1'000'000.0 << "MB" << std::endl;
#else
    return;
#endif
}

template <typename SourceView, typename TargetView>
inline void SafeDeepCopy(TargetView& tgt, const SourceView& src)
{
    using SrcExecSpace = typename SourceView::execution_space;
    using TgtExecSpace = typename TargetView::execution_space;
    using TgtLayout = typename TargetView::array_layout;

    const auto mirror =
        Kokkos::create_mirror_view_and_copy(SrcExecSpace{}, src);
    Kokkos::deep_copy(tgt, mirror);
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

template <typename DataView>
void export_mat_view(const DataView& data, const int rows, const int cols,
                     std::fstream& file)
{
    using layout = typename DataView::array_layout;
    Kokkos::View<typename DataView::const_value_type**, layout,
                 Kokkos::DefaultHostExecutionSpace>
        M(data.data(), rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols - 1; ++j)
        {
            file << M(i, j) << ",";
        }
        file << M(i, cols - 1) << "\n";
    }
    file.flush();
}

template <typename ViewType, typename Comparator>
KOKKOS_INLINE_FUNCTION auto MyMinMax(const ViewType& v, const Comparator& op)
{
    using T = typename ViewType::non_const_value_type;
    using R = Kokkos::pair<T, T>;
    const int n = v.extent_int(0);
    KOKKOS_ASSERT(n > 0);
    R ret = Kokkos::make_pair(v(0), v(0));
    for (int i = 1; i < n; ++i)
    {
        const auto elt = v(i);
        if (op(elt, ret.first))
        {
            ret.first = elt;
        }
        if (op(ret.second, elt))
        {
            ret.second = elt;
        }
    }
    return ret;
}
