cmake_minimum_required(VERSION 3.31)

# Default to Release build if no build type is specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified — defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

project(PACMAN VERSION 0.1.0 LANGUAGES CXX DESCRIPTION "Portable Field Transfer (PFT) Library for Interpolating Large-Scale Scientific Data Sets")

# -----------------------------------------------------
# Optional Fortran support — must be enabled before subdirectories
# so that enable_language(Fortran) is available to all targets.
# -----------------------------------------------------
option(BUILD_FORTRAN_INTERFACE "Build the Fortran interface module and test" OFF)
if (BUILD_FORTRAN_INTERFACE)
    include(CheckLanguage)
    check_language(Fortran)
    if (CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
        message(STATUS "Fortran compiler found: ${CMAKE_Fortran_COMPILER}")
    else()
        message(FATAL_ERROR
            "BUILD_FORTRAN_INTERFACE is ON but no Fortran compiler was found. "
            "Install a Fortran compiler (e.g. gfortran) or set -DBUILD_FORTRAN_INTERFACE=OFF.")
    endif()
endif()

# -----------------------------------------------------
# Global configuration for the project
# -----------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable -fPIC for all targets")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Disable -flto for all targets")
set(EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands")

set(CMAKE_SHARED_LIBRARY_ENABLE_EXPORTS ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

option(BUILD_SHARED_LIBS "Build shared libraries (.so/.dylib) instead of static (.a)" ON)
option(BUILD_PYTHON_INTERFACE "Build the Python interface module and test" ON)
option(BUILD_TESTS "Configure tests using ctest" ON)
option(ENABLE_COVERAGE "Enable coverage tests (requires BUILD_TESTS=ON)" OFF)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -----------------------------------------------------
# Library dependencies
# -----------------------------------------------------
find_package(Kokkos 5.0 REQUIRED)
find_package(KokkosKernels 5.0 REQUIRED)
find_package(ArborX 2.1 REQUIRED)

# -----------------------------------------------------
# Main Target (Interface) + optionnal Python bindings
# -----------------------------------------------------
add_subdirectory(src)

# -----------------------------------------------------
# Test executable (require VTK)
# -----------------------------------------------------
if (BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------
# Package config files
# -----------------------------------------------------
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PACMANConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PACMANConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PACMANConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PACMAN
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PACMANConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PACMANConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PACMAN
)
