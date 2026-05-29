//
// This file is subject to the terms and conditions defined in
// file 'LICENSE', which is part of this source code package.
//

#pragma once

// <pybind11/pybind11.h> includes "Python.h"
// which must be included first (CPython requirement)

// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// clang-format on

#include <Kokkos_Core.hpp>
#include <iostream>
#include <optional>

namespace PACMAN {
namespace py = pybind11;

template <typename T>
using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T> using optional_np_array = std::optional<np_array<T>>;

/// @brief Macro used to set the execution space values into a Python submodule
#define ADD_EXECSPACES_ENTRY(E) execspaces.attr(#E) = ExecSpaces::E

/// @brief A struct which represents the execution spaces from Kokkos using
/// integer values to receive it easily from Python
/// @note Struct entries underlying integer values are meaningless and can be
/// changed
struct ExecSpaces {
  static constexpr const unsigned char SERIAL = 0x00;
  static constexpr const unsigned char OPENMP = 0x01;
  static constexpr const unsigned char THREADS = 0x02;
  static constexpr const unsigned char CUDA = 0x03;
  static constexpr const unsigned char HIP = 0x04;
  static constexpr const unsigned char SYCL = 0x05;
};

// clang-format off

/// @brief A type variant which defines the enabled Kokkos execution spaces, this variant is mandatory to pass the execution space as a template argument to the transfer method function.
using AvailableExecSpaces = std::variant<
    #if defined(KOKKOS_ENABLE_SERIAL)
        Kokkos::Serial
    #endif
    #if defined(KOKKOS_ENABLE_OPENMP)
    ,
        Kokkos::OpenMP
    #endif
    #if defined(KOKKOS_ENABLE_THREADS)
    ,
        Kokkos::Threads
    #endif
    #if defined(KOKKOS_ENABLE_CUDA)
    ,
        Kokkos::Cuda
    #endif
    #if defined(KOKKOS_ENABLE_HIP)
    ,
        Kokkos::HIP
    #endif
    #if defined(KOKKOS_ENABLE_SYCL)
    ,
        Kokkos::SYCL
    #endif
>;
// clang-format on

/// @brief Converts an execution space from an unsigned char to a variant type
static inline AvailableExecSpaces MakeExecSpace(const unsigned char s) {
  // clang-format off
        switch (s) {
            case ExecSpaces::SERIAL:
                #if defined(KOKKOS_ENABLE_SERIAL)
                    return Kokkos::Serial{};
                #else
                    std::cerr << "Kokkos::Serial is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SERIAL!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            case ExecSpaces::OPENMP:
                #if defined(KOKKOS_ENABLE_OPENMP)
                    return Kokkos::OpenMP{};
                #else
                    std::cerr << "Kokkos::OpenMP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_OPENMP!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            case ExecSpaces::THREADS:
                #if defined(KOKKOS_ENABLE_THREADS)
                    return Kokkos::Threads{};
                #else
                    std::cerr << "Kokkos::Threads is not enabled! Please recompile Kokkos with Kokkos_ENABLE_THREADS!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            case ExecSpaces::CUDA:
                #if defined(KOKKOS_ENABLE_CUDA)
                    return Kokkos::Cuda{};
                #else
                    std::cerr << "Kokkos::Cuda is not enabled! Please recompile Kokkos with Kokkos_ENABLE_CUDA!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            case ExecSpaces::HIP:
                #if defined(KOKKOS_ENABLE_HIP)
                    return Kokkos::HIP{};
                #else
                    std::cerr << "Kokkos::HIP is not enabled! Please recompile Kokkos with Kokkos_ENABLE_HIP!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            case ExecSpaces::SYCL:
                #if defined(KOKKOS_ENABLE_SYCL)
                    return Kokkos::SYCL{};
                #else
                    std::cerr << "Kokkos::SYCL is not enabled! Please recompile Kokkos with Kokkos_ENABLE_SYCL!" << std::endl;
                    exit(EXIT_FAILURE);
                #endif
            default:
                std::cerr << "execspace: Serial, OpenMP, Threads, Cuda, HIP, SYCL"
                        << std::endl;
                exit(EXIT_FAILURE);
        }
  // clang-format on
}
} // namespace PACMAN
