# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-01 (Initial release of the package)

### Added
- Initial public release of PACMAN, a portable field-transfer library written in C++20 with Kokkos for on-node performance portability.
- Core algorithm libraries (rbfpum, finiteelements) are header-only Kokkos/C++ template libraries requiring C++20 and Kokkos ≥ 5.0 / KokkosKernels ≥ 5.0 / ArborX ≥ 2.1.
- Compiled C++ interface library PACMAN::pacman_interface_cpp with explicit template instantiations (ETI) for every supported Kokkos execution space (Serial, OpenMP, Threads, CUDA, HIP, SYCL), enabling link-time use without recompiling templates.
- Optional compiled Fortran interface library PACMAN::pacman_interface_fortran (enabled via -DBUILD_FORTRAN_INTERFACE=ON), providing a Fortran module (pacman_fortran) that wraps the C++ interface for use from Fortran callers.
- Python bindings (enabled via -DBUILD_MODULE=ON, default ON) via pybind11 for both FE and RBF-PUM interpolation APIs, also built with ETI for each execution space.
- Public CMake package targets exported via PACMANTargets.cmake:
PACMAN::PACMAN — header-only aggregate interface target (links core + Kokkos + KokkosKernels + ArborX).
PACMAN::pacman_interface_cpp — compiled C++ flat-API library.
PACMAN::pacman_interface_fortran — compiled Fortran module library (when BUILD_FORTRAN_INTERFACE=ON).
- Finite-elements interpolation methods:
NEAREST_NEAREST
INTERP_CLAMP
INTERP_NEAREST
INTERP_ZEROFILL
INTERP_EXTRAP
- RBF-PUM interpolation implementation with Wendland kernels (C0, C2, C4, C6, C8).
- ArborX -MLS interpolation.
Execution-space selection constants for all supported Kokkos backends (Serial, OpenMP, Threads, CUDA, HIP, SYCL).
- CMake test integration (-DBUILD_TESTS=ON) with C++, Fortran, and Python functional tests, including mesh-based interpolation datasets in tests/meshes/.
- Optional coverage instrumentation via -DENABLE_COVERAGE=ON.

### Notes
- This is the first tagged release series (v0.1.0) and establishes the initial public API baseline.
- The project builds shared libraries by default (BUILD_SHARED_LIBS=ON); static builds are supported via -DBUILD_SHARED_LIBS=OFF.
- Tested localy for Serial, OpenMP, Threads, CUDA and HIP.