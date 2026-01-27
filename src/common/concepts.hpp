#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <type_traits>

namespace PACMAN {
/* Type Evaluation */

template <typename> struct is_kokkos_view : std::false_type {};

template <typename DataType, typename... P>
struct is_kokkos_view<Kokkos::View<DataType, P...>> : std::true_type {};

template <typename> struct is_kokkos_array : std::false_type {};

template <typename T, size_t N>
struct is_kokkos_array<Kokkos::Array<T, N>> : std::true_type {};

/* Type Constraints */

template <typename T>
concept IsKokkosView = is_kokkos_view<std::remove_cv_t<T>>::value;

template <typename T, int R>
concept IsRank = std::remove_cv_t<T>::rank == R;

template <class T>
concept IsHostRBFFunction = requires(const T t, const double x) {
  { t.EvalHost(x) } -> std::floating_point;
};

template <class T>
concept IsRBFFunction = requires(const T t, const double x) {
  { t.Eval(x) } -> std::floating_point;
};

template <class T>
concept IsKokkosArray = is_kokkos_array<std::remove_cv_t<T>>::value;

template <int_t Dim>
concept IsValidDim = (Dim >= 1 && Dim <= 3);

/* Concepts as type */

template <typename T, int R>
concept KokkosViewRank = IsKokkosView<T> && IsRank<T, R>;

template <typename T>
concept KokkosArray = IsKokkosArray<T>;

template <typename T>
concept HostRBFFunction = IsHostRBFFunction<T>;

template <typename T>
concept RBFFunction = IsRBFFunction<T>;

} // namespace PACMAN
