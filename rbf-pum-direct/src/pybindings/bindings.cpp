#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <chrono>
#include <interpolator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pybindings/utils.hpp"

namespace py = pybind11;
using PyBindsFPType = double;
using np_array =
    py::array_t<PyBindsFPType, py::array::c_style | py::array::forcecast>;

template <int Dim, typename ArrType>
inline auto get_point(const ArrType& arr, const int i)
{
    auto p = ArborX::Point<Dim, PyBindsFPType>{};
    for (int d = 0; d < Dim; ++d)
    {
        p[d] = arr(i, d);
    }
    return p;
}

template <typename ExecSpace, typename RbfFuncType, int Dim>
py::array_t<PyBindsFPType> run_interpolate(np_array& source_points,
                                           np_array& source_values,
                                           np_array& target_points)
{
    const int N = source_points.shape(0);
    const int M = target_points.shape(0);
    using Point = ArborX::Point<Dim, PyBindsFPType>;
    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    using PointsVectorView = Kokkos::View<Point*, HostSpace>;
    using ScalarVectorView = Kokkos::View<PyBindsFPType*, HostSpace>;

    PointsVectorView kk_source_points(
        Kokkos::view_alloc(HostSpace{}, Kokkos::WithoutInitializing,
                           "run_interpolate::kk_source_points"),
        N);
    ScalarVectorView kk_source_values(
        Kokkos::view_alloc(HostSpace{}, Kokkos::WithoutInitializing,
                           "run_interpolate::kk_source_values"),
        N);
    PointsVectorView kk_target_points(
        Kokkos::view_alloc(HostSpace{}, Kokkos::WithoutInitializing,
                           "run_interpolate::kk_target_points"),
        M);

    auto unchecked_source_points = source_points.unchecked<2>();
    auto unchecked_source_values = source_values.unchecked<1>();
    auto unchecked_target_points = target_points.unchecked<2>();

    Kokkos::parallel_for(
        "run_interpolate::p_for fill input views",
        Kokkos::RangePolicy(HostSpace{}, 0, N), [&](const int& i) {
            kk_source_points(i) = get_point<Dim>(unchecked_source_points, i);
            kk_source_values(i) = unchecked_source_values(i);
        });
    Kokkos::parallel_for(
        "run_interpolate::p_for fill target view",
        Kokkos::RangePolicy(HostSpace{}, 0, M), [&](const int& i) {
            kk_target_points(i) = get_point<Dim>(unchecked_target_points, i);
        });
    Kokkos::fence();

    RbfFuncType rbf_function{};
    const auto src_pts =
        Kokkos::create_mirror_view_and_copy(ExecSpace{}, kk_source_points);
    const auto src_val =
        Kokkos::create_mirror_view_and_copy(ExecSpace{}, kk_source_values);
    const auto tgt_pts =
        Kokkos::create_mirror_view_and_copy(ExecSpace{}, kk_target_points);
    RbfPumInterpolator<ExecSpace, Dim, PyBindsFPType, RbfFuncType> interpolator(
        src_pts, src_val, tgt_pts, rbf_function);

    const auto out =
        Kokkos::create_mirror_view_and_copy(HostSpace{}, interpolator.out);
    auto result = py::array_t<PyBindsFPType>(out.size());
    auto mutable_unchecked_res = result.mutable_unchecked<1>();

    Kokkos::parallel_for(
        "run_interpolate::p_for fill out array",
        Kokkos::RangePolicy(HostSpace{}, 0, M),
        [&](const int& i) { mutable_unchecked_res(i) = out(i); });
    Kokkos::fence();

    return result;
}

template <int Dim>
py::array_t<PyBindsFPType>
dispatch(np_array& source_points, np_array& source_values,
         np_array& target_points, const char execspace, const char rbf_function)
{
    return std::visit(
        [&](auto execspace_t, auto rbf_function_t) {
            return run_interpolate<decltype(execspace_t),
                                   decltype(rbf_function_t), Dim>(
                source_points, source_values, target_points);
        },
        make_execspace(execspace),
        make_rbffunction<PyBindsFPType>(rbf_function));
}

py::array_t<PyBindsFPType> interpolate(np_array source_points,
                                       np_array source_values,
                                       np_array target_points, char execspace,
                                       char rbf_function)
{
    const std::string _region_name = "pybinds::interpolate";
    const Kokkos::Profiling::ScopedRegion region(_region_name);

    if (source_points.ndim() != 2)
    {
        throw std::runtime_error("source_points' shape must be (N_src, dim)");
    }
    if (target_points.ndim() != 2)
    {
        throw std::runtime_error("target_points' shape must be (N_tgt, dim)");
    }
    if (source_points.shape(1) != target_points.shape(1))
    {
        throw std::runtime_error(
            "source points dims must match target points dims");
    }
    if (source_points.shape(0) != source_values.shape(0))
    {
        throw std::runtime_error(
            "each source point must have its corresponding value in the "
            "source_values np.array");
    }

    const int dim = source_points.shape(1);
    switch (dim)
    {
    case 1:
        return dispatch<1>(source_points, source_values, target_points,
                           execspace, rbf_function);
    case 2:
        return dispatch<2>(source_points, source_values, target_points,
                           execspace, rbf_function);
    case 3:
        return dispatch<3>(source_points, source_values, target_points,
                           execspace, rbf_function);
    default:
        throw new std::runtime_error(
            "the points dimensions must be only 1, 2 or 3 for now");
    }
}

// clang-format off
PYBIND11_MODULE(interpolator, m)
{
    if (!Kokkos::is_initialized())
        Kokkos::initialize();
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

    // Common execspaces attributes
    auto execspaces = m.def_submodule("execspaces", "Available execution spaces values for Kokkos");
    ADD_EXECSPACES_ENTRY(SERIAL);
    ADD_EXECSPACES_ENTRY(OPENMP);
    ADD_EXECSPACES_ENTRY(THREADS);
    ADD_EXECSPACES_ENTRY(CUDA);
    ADD_EXECSPACES_ENTRY(HIP);
    ADD_EXECSPACES_ENTRY(SYCL);

    auto rbf = m.def_submodule("rbf", "Radial-based functions partition of unity method interpolator (CPU/GPU)");
    rbf.def("interpolate",          // Python Func Name
        &interpolate,               // C++ Corresponding Function
        py::arg("source_points"),   // #1 Arg
        py::arg("source_values"),   // #2 Arg
        py::arg("target_points"),   // #3 Arg
        py::arg("execspace"),       // #4 Arg
        py::arg("rbf_function")     // #5 Arg
    );

    auto rbf_functions = rbf.def_submodule("functions", "Available RBF functions for the RBF-PUM interpolator");
    ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC0);
    ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC2);
    ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC4);
    ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC6);
    ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC8);
}
// clang-format on