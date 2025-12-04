#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <chrono>
#include <interpolator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using PyBindsFPType = double;
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
using ScalarVectorView = Kokkos::View<PyBindsFPType*, ExecSpace>;
using np_array =
    py::array_t<PyBindsFPType, py::array::c_style | py::array::forcecast>;

template <int Dim, typename ArrType>
__forceinline__ auto get_point(const ArrType& arr, const int i)
{
    if constexpr (Dim == 2)
    {
        return ArborX::Point<3, PyBindsFPType>{ arr(i, 0), arr(i, 1), 0.0 };
    }
    if constexpr (Dim == 3)
    {
        return ArborX::Point<3, PyBindsFPType>{ arr(i, 0), arr(i, 1),
                                                arr(i, 2) };
    }
    return ArborX::Point<3, PyBindsFPType>{};
}

template <typename ArrType>
__forceinline__ auto get_2d_point(const ArrType& arr, const int i)
{
    return ArborX::Point<2, PyBindsFPType>{ arr(i, 0), arr(i, 1) };
}

py::array_t<PyBindsFPType> _2d_interpolate(np_array& source_points,
                                           np_array& source_values,
                                           np_array& target_points)
{
    const Kokkos::Profiling::ScopedRegion region("pybinds::_2d_interpolate");
    constexpr int Dim = 2;
    const int N = source_points.shape(0);
    const int M = target_points.shape(0);

    using Point = ArborX::Point<Dim, PyBindsFPType>;
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    using ScalarVectorView = Kokkos::View<PyBindsFPType*, ExecSpace>;
    using PointsVectorView = Kokkos::View<Point*, ExecSpace>;

    PointsVectorView kk_source_points(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_source_points"),
        N);
    ScalarVectorView kk_source_values(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_source_values"),
        N);
    PointsVectorView kk_target_points(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_target_points"),
        M);

    auto unchecked_source_points = source_points.unchecked<2>();
    auto unchecked_source_values = source_values.unchecked<1>();
    auto unchecked_target_points = target_points.unchecked<2>();

    Kokkos::parallel_for(
        "pybinds::p_for fill input views",
        Kokkos::RangePolicy(ExecSpace{}, 0, N), [&](const int& i) {
            kk_source_points(i) = get_2d_point(unchecked_source_points, i);
            kk_source_values(i) = unchecked_source_values(i);
        });
    Kokkos::parallel_for(
        "pybinds::p_for fill target view",
        Kokkos::RangePolicy(ExecSpace{}, 0, M), [&](const int& i) {
            kk_target_points(i) = get_2d_point(unchecked_target_points, i);
        });
    Kokkos::fence();

    WendlandC6<PyBindsFPType> rbf_function;
    RbfPumInterpolator<ExecSpace, Dim, PyBindsFPType, WendlandC6<PyBindsFPType>>
        interpolator(kk_source_points, kk_source_values, kk_target_points,
                     rbf_function);
    Kokkos::fence();

    ScalarVectorView out = interpolator.out;
    auto result = py::array_t<PyBindsFPType>(out.size());
    auto mutable_unchecked_res = result.mutable_unchecked<1>();

    Kokkos::parallel_for(
        "pybinds::p_for fill out array", Kokkos::RangePolicy(ExecSpace{}, 0, M),
        [&](const int& i) { mutable_unchecked_res(i) = out(i); });
    Kokkos::fence();

    return result;
}

template <int Dim>
py::array_t<PyBindsFPType> _interpolate_internal(np_array& source_points,
                                                 np_array& source_values,
                                                 np_array& target_points)
{
    const Kokkos::Profiling::ScopedRegion region(
        "pybinds::_interpolate_internal");
    const int N = source_points.shape(0);
    const int M = target_points.shape(0);

    using Point = ArborX::Point<3, PyBindsFPType>;
    using PointsVectorView = Kokkos::View<Point*, ExecSpace>;

    PointsVectorView kk_source_points(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_source_points"),
        N);
    ScalarVectorView kk_source_values(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_source_values"),
        N);
    PointsVectorView kk_target_points(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                           "pybinds::kk_target_points"),
        M);

    auto unchecked_source_points = source_points.unchecked<2>();
    auto unchecked_source_values = source_values.unchecked<1>();
    auto unchecked_target_points = target_points.unchecked<2>();

    Kokkos::parallel_for(
        "pybinds::p_for fill input views",
        Kokkos::RangePolicy(ExecSpace{}, 0, N), [&](const int& i) {
            kk_source_points(i) = get_point<Dim>(unchecked_source_points, i);
            kk_source_values(i) = unchecked_source_values(i);
        });
    Kokkos::parallel_for(
        "pybinds::p_for fill target view",
        Kokkos::RangePolicy(ExecSpace{}, 0, M), [&](const int& i) {
            kk_target_points(i) = get_point<Dim>(unchecked_target_points, i);
        });
    Kokkos::fence();

    WendlandC6<PyBindsFPType> rbf_function;
    RbfPumInterpolator<ExecSpace, 3, PyBindsFPType, WendlandC6<PyBindsFPType>>
        interpolator(kk_source_points, kk_source_values, kk_target_points,
                     rbf_function);

    ScalarVectorView out = interpolator.out;
    auto result = py::array_t<PyBindsFPType>(out.size());
    auto mutable_unchecked_res = result.mutable_unchecked<1>();

    Kokkos::parallel_for(
        "pybinds::p_for fill out array", Kokkos::RangePolicy(ExecSpace{}, 0, M),
        [&](const int& i) { mutable_unchecked_res(i) = out(i); });
    Kokkos::fence();

    return result;
}

py::array_t<PyBindsFPType> interpolate(np_array source_points,
                                       np_array source_values,
                                       np_array target_points)
{
    const Kokkos::Profiling::ScopedRegion region("pybinds::interpolate");
    auto buf_src = source_points.request();
    auto buf_val = source_values.request();
    auto buf_tgt = target_points.request();

    if (buf_src.ndim != 2)
        throw std::runtime_error("source_points must be (N_src, dim)");
    if (buf_tgt.ndim != 2)
        throw std::runtime_error("target_points must be (N_tgt, dim)");
    if (buf_src.shape[1] != buf_tgt.shape[1])
    {
        throw std::runtime_error(
            "source points dims must match target points dims");
    }
    if (buf_src.shape[0] != buf_val.shape[0])
    {
        throw std::runtime_error(
            "each source point must have its corresponding value in the "
            "source_values np.array");
    }

    const int dim = buf_src.shape[1];

    switch (dim)
    {
    case 2:
        return _interpolate_internal<2>(source_points, source_values,
                                        target_points);
    case 3:
        return _interpolate_internal<3>(source_points, source_values,
                                        target_points);
    default:
        throw new std::runtime_error(
            "the points dimensions must be only 2 or 3 for now");
    }
}

py::array_t<PyBindsFPType> interpolate_2d(np_array source_points,
                                          np_array source_values,
                                          np_array target_points)
{
    const Kokkos::Profiling::ScopedRegion region("pybinds::interpolate_2d");
    auto buf_src = source_points.request();
    auto buf_val = source_values.request();
    auto buf_tgt = target_points.request();

    if (buf_src.ndim != 2)
        throw std::runtime_error("source_points must be (N_src, dim)");
    if (buf_tgt.ndim != 2)
        throw std::runtime_error("target_points must be (N_tgt, dim)");
    if (buf_src.shape[1] != buf_tgt.shape[1])
    {
        throw std::runtime_error(
            "source points dims must match target points dims");
    }
    if (buf_src.shape[0] != buf_val.shape[0])
    {
        throw std::runtime_error(
            "each source point must have its corresponding value in the "
            "source_values np.array");
    }
    return _2d_interpolate(source_points, source_values, target_points);
}

// clang-format off
PYBIND11_MODULE(RbfPumInterpolator, m)
{
    if (!Kokkos::is_initialized())
        Kokkos::initialize();
    m.def("interpolate",          // Python Func Name
        &interpolate,             // C++ Corresponding Function
        py::arg("source_points"), // #1 Arg
        py::arg("source_values"), // #2 Arg
        py::arg("target_points")  // #3 Arg
    );
    std::atexit([](){
        if (Kokkos::is_initialized() && !Kokkos::is_finalized())
            Kokkos::finalize();
    });

    // Manage Kokkos state using python calls
    m.def("initialize",
    []() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    });
    m.def("finalize", []() {
        if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
            Kokkos::finalize();
        }
    });

    // debug func
        m.def("interpolate_2d",
        &interpolate_2d,
        py::arg("source_points"),
        py::arg("source_values"),
        py::arg("target_points")
    );
}
// clang-format on