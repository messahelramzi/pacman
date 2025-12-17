cmake_minimum_required(VERSION 3.31)

project(RBFPUMInterpolator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Required by Kokkos
set(CMAKE_CXX_EXTENSIONS OFF)
# Required for the Python Module
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable -fPIC for all targets")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
# Useful to configure the IDE
set(EXPORT_COMPILE_COMMANDS ON)

option(BUILD_MODULE "Build the Python module" ON)
option(BUILD_TESTS "Build the test executable" OFF)

# -----------------------------------------------------
# Main Target (Interface)
# -----------------------------------------------------
find_package(Kokkos 5.0 REQUIRED)
find_package(ArborX 2.0 REQUIRED)
find_package(KokkosKernels 5.0 REQUIRED)
add_library(RbfPumInterpolator INTERFACE)
target_link_libraries(RbfPumInterpolator INTERFACE
                        Kokkos::kokkos
                        KokkosKernels::kokkoskernels
                        ArborX::ArborX)
target_include_directories(RbfPumInterpolator INTERFACE src)
target_compile_features(RbfPumInterpolator INTERFACE cxx_std_20)

# -----------------------------------------------------
# Test executable (require VTK)
# -----------------------------------------------------
if (BUILD_TESTS)
    find_package(VTK COMPONENTS
        CommonCore
        IOXML
        CommonDataModel
        REQUIRED)
    add_executable(test tests/tests.cpp)
    target_link_libraries(test PRIVATE
        RbfPumInterpolator
        VTK::CommonCore
        VTK::IOXML
        VTK::CommonDataModel
    )
    target_include_directories(test PRIVATE "src/")
    target_include_directories(test PRIVATE "tests/")
    target_compile_features(test PRIVATE cxx_std_20)
    target_compile_options(test PRIVATE
        -m64
        -lineinfo # for HPCToolKit profiler
    )
endif()

# -----------------------------------------------------
# Python Module (Requires PyBind11)
# -----------------------------------------------------
if (BUILD_MODULE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(interpolator "src/pybindings/bindings.cpp")
    target_link_libraries(interpolator PRIVATE RbfPumInterpolator)
    target_include_directories(interpolator PRIVATE "src")
endif()
