#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>

#include "utils.hpp"

template <int Dim, class Coordinates>
struct KNearest
{
    int k;
    ArborX::Point<Dim, Coordinates> random_point;
};

template <int Dim, class Coordinates>
struct ArborX::AccessTraits<KNearest<Dim, Coordinates>>
{
    using memory_space = Kokkos::CudaSpace::memory_space;
    static KOKKOS_FUNCTION std::size_t
    size(const KNearest<Dim, Coordinates>& /* kn */)
    {
        return 1;
    }
    static KOKKOS_FUNCTION auto get(const KNearest<Dim, Coordinates>& kn,
                                    std::size_t /* i */)
    {
        return ArborX::nearest(kn.random_point, kn.k);
    }
};

template <int Dim, class Coordinates>
struct KNearestCallback
{
    ArborX::Point<Dim, Coordinates> center;
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate, Value const& value,
                                    OutputFunctor const& out) const
    {
        out(_NDdistance_without_sqrt<Dim, Coordinates>(center, value.value));
    }
};

#endif /* ! CALLBACKS_HPP */
