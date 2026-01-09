#include "pybindings/bindings.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "common/types.hpp"
#include "pybindings/fe_bindings.hpp"
#include "pybindings/rbf_bindings.hpp"

namespace PACMAN {
PYBIND11_MODULE(pacman, m) {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  std::atexit([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  });

  // Manage Kokkos state using python calls
  m.def("initialize", []() {
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
  auto execspaces = m.def_submodule(
      "execspaces", "Available execution spaces values for Kokkos");
  ADD_EXECSPACES_ENTRY(SERIAL);
  ADD_EXECSPACES_ENTRY(OPENMP);
  ADD_EXECSPACES_ENTRY(THREADS);
  ADD_EXECSPACES_ENTRY(CUDA);
  ADD_EXECSPACES_ENTRY(HIP);
  ADD_EXECSPACES_ENTRY(SYCL);

  auto rbf = m.def_submodule("rbf", "Radial-based functions partition of unity "
                                    "method interpolator (CPU/GPU)");
  rbf.def("interpolate",               // Python Func Name
          &PyBindingsRbf::Interpolate, // C++ Corresponding Function
          py::arg("execspace"),        // #1 Arg
          py::arg("rbf_function"),     // #2 Arg
          py::arg("source_points"),    // #3 Arg
          py::arg("source_values"),    // #4 Arg
          py::arg("target_points")     // #5 Arg
  );

  auto rbf_functions = rbf.def_submodule(
      "functions", "Available RBF functions for the RBF-PUM interpolator");
  ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC0);
  ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC2);
  ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC4);
  ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC6);
  ADD_RBF_FUNCTIONS_ENTRY(WENDLANDC8);

  auto fe = m.def_submodule(
      "fe", "Finite elements interpolation functions (CPU/GPU)");
  fe.def("interpolate", &PyBindingsFiniteElements::Interpolate,
         py::arg("execspace"),     // #1 Arg
         py::arg("method"),        // #2 Arg
         py::arg("source_points"), // #3 Arg
         py::arg("source_values"), // #4 Arg
         py::arg("conn_val"),      // #5 Arg
         py::arg("conn_off"),      // #6 Arg
         py::arg("cell_types"),    // #7 Arg
         py::arg("target_points")  // #8 Arg
  );
  auto fe_methods = fe.def_submodule(
      "methods", "Available FE methods for the Finite Elements interpolator");
  ADD_FE_METHODS_ENTRY(NEAREST_NEAREST);
  ADD_FE_METHODS_ENTRY(INTERP_CLAMP);
  ADD_FE_METHODS_ENTRY(INTERP_NEAREST);
  ADD_FE_METHODS_ENTRY(INTERP_ZEROFILL);
  ADD_FE_METHODS_ENTRY(INTERP_EXTRAP);
}

} // namespace PACMAN
