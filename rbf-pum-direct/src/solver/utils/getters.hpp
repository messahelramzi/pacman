#pragma once

#include <Kokkos_Core.hpp>

#include "concepts.hpp"
#include "utils/utils.hpp"

template <KokkosViewRank<1> AsViewType, KokkosViewRank<1> OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfMatrix(const int& k, const AsViewType& As, const OffsViewType& offs,
             const int& n, const int& m)
{
    using data_type = typename AsViewType::non_const_value_type;
    using memory_space = typename AsViewType::memory_space;
    const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
    auto A_data = Kokkos::subview(As, slice);
    return Kokkos::View<data_type**, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(A_data.data(),
                                                                 n, m);
}

template <KokkosViewRank<1> PsViewType, KokkosViewRank<1> OffsViewType,
          KokkosArray AxisType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetPolyMatrix(const int& k, const PsViewType& Ps, const OffsViewType& offs,
              const int& n, const AxisType& active_axis)
{
    using data_type = typename PsViewType::non_const_value_type;
    using memory_space = typename PsViewType::memory_space;
    const int N = offs(k + 1) - offs(k);
    int M = active_axis.size() + 1;
    const auto slice = Kokkos::make_pair(offs(k) * M, offs(k + 1) * M);
    auto P_data = Kokkos::subview(Ps, slice);
    M = 1;
    for (int d = 0; d < active_axis.size(); ++d)
    {
        if (active_axis[d])
        {
            ++M;
        }
    }

    return Kokkos::View<data_type**, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(P_data.data(),
                                                                 n, M);
}

template <KokkosViewRank<1> BsViewType, KokkosViewRank<1> OffsViewType>
KOKKOS_FORCEINLINE_FUNCTION auto
GetRbfVector(const int& k, const BsViewType& Bs, const OffsViewType& offs,
             const int& n)
{
    using data_type = typename BsViewType::non_const_value_type;
    using memory_space = typename BsViewType::memory_space;
    const auto slice = Kokkos::make_pair(offs(k), offs(k + 1));
    auto B_data = Kokkos::subview(Bs, slice);
    return Kokkos::View<data_type*, memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(B_data.data(),
                                                                 n);
}
