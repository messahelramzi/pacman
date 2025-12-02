#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <chrono>
#include <interpolator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using My_PyBinds_ScalarType = double;

using np_array = py::array_t<My_PyBinds_ScalarType,
                             py::array::c_style | py::array::forcecast>;

py::array_t<My_PyBinds_ScalarType> interpolate_2d(np_array& source_points,
                                                  np_array& source_values,
                                                  np_array& target_points)
{
    const Kokkos::Profiling::ScopedRegion region("pybinds::interpolate_2d");
    constexpr int Dim = 2;
    const int N = source_points.shape(0);
    const int M = target_points.shape(0);

    using Point = ArborX::Point<Dim, My_PyBinds_ScalarType>;
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    using ScalarVectorView = Kokkos::View<My_PyBinds_ScalarType*, ExecSpace>;
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

    Kokkos::parallel_for("pybinds::p_for fill input views",
                         Kokkos::RangePolicy(ExecSpace{}, 0, N),
                         [&](const int& i) {
                             kk_source_points(i) = Point{
                                 unchecked_source_points(i, 0),
                                 unchecked_source_points(i, 1),
                             };
                             kk_source_values(i) = unchecked_source_values(i);
                         });
    Kokkos::parallel_for("pybinds::p_for fill target view",
                         Kokkos::RangePolicy(ExecSpace{}, 0, M),
                         [&](const int& i) {
                             kk_target_points(i) = Point{
                                 unchecked_target_points(i, 0),
                                 unchecked_target_points(i, 1),
                             };
                         });
    Kokkos::fence();

    WendlandC6<My_PyBinds_ScalarType> rbf_function;
    RbfPumInterpolator<ExecSpace, Dim, My_PyBinds_ScalarType,
                       WendlandC6<My_PyBinds_ScalarType>>
        interpolator(kk_source_points, kk_source_values, kk_target_points,
                     rbf_function);

    ScalarVectorView out = interpolator.out;
    auto result = py::array_t<My_PyBinds_ScalarType>(out.size());
    auto mutable_unchecked_res = result.mutable_unchecked<1>();

    Kokkos::parallel_for(
        "pybinds::p_for fill out array", Kokkos::RangePolicy(ExecSpace{}, 0, M),
        [&](const int& i) { mutable_unchecked_res(i) = out(i); });
    Kokkos::fence();

    return result;
}

py::array_t<My_PyBinds_ScalarType> interpolate_3d(np_array& source_points,
                                                  np_array& source_values,
                                                  np_array& target_points)
{
    const Kokkos::Profiling::ScopedRegion region("pybinds::interpolate_3d");
    constexpr int Dim = 3;
    const int N = source_points.shape(0);
    const int M = target_points.shape(0);

    using Point = ArborX::Point<Dim, My_PyBinds_ScalarType>;
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    using ScalarVectorView = Kokkos::View<My_PyBinds_ScalarType*, ExecSpace>;
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
            kk_source_points(i) = Point{ unchecked_source_points(i, 0),
                                         unchecked_source_points(i, 1),
                                         unchecked_source_points(i, 2) };
            kk_source_values(i) = unchecked_source_values(i);
        });
    Kokkos::parallel_for(
        "pybinds::p_for fill target view",
        Kokkos::RangePolicy(ExecSpace{}, 0, M), [&](const int& i) {
            kk_target_points(i) = Point{ unchecked_target_points(i, 0),
                                         unchecked_target_points(i, 1),
                                         unchecked_target_points(i, 2) };
        });
    Kokkos::fence();

    WendlandC6<My_PyBinds_ScalarType> rbf_function;
    RbfPumInterpolator<ExecSpace, Dim, My_PyBinds_ScalarType,
                       WendlandC6<My_PyBinds_ScalarType>>
        interpolator(kk_source_points, kk_source_values, kk_target_points,
                     rbf_function);

    ScalarVectorView out = interpolator.out;
    auto result = py::array_t<My_PyBinds_ScalarType>(out.size());
    auto mutable_unchecked_res = result.mutable_unchecked<1>();

    Kokkos::parallel_for(
        "pybinds::p_for fill out array", Kokkos::RangePolicy(ExecSpace{}, 0, M),
        [&](const int& i) { mutable_unchecked_res(i) = out(i); });
    Kokkos::fence();

    return result;
}

py::array_t<My_PyBinds_ScalarType> interpolate(np_array source_points,
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
        return interpolate_2d(source_points, source_values, target_points);
    case 3:
        return interpolate_3d(source_points, source_values, target_points);
    default:
        throw new std::runtime_error(
            "the points dimensions must be only 2 or 3 for now");
    }
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
}
// clang-format on