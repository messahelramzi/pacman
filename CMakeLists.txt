cmake_minimum_required(VERSION 3.31)

project(PACMAN VERSION 0.1.0 LANGUAGES CXX DESCRIPTION "Portable Field Transfer (PFT) Library for Interpolating Large-Scale Scientific Data Sets")

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

option(BUILD_MODULE "Build the Python module" ON)
option(BUILD_TESTS "Configure tests using ctest" ON)
option(ENABLE_COVERAGE "Enable coverage tests (requires BUILD_TESTS=ON)" ON)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -----------------------------------------------------
# Library dependencies
# -----------------------------------------------------
find_package(Kokkos 5.0 REQUIRED)
find_package(ArborX 2.0 REQUIRED)
find_package(KokkosKernels 5.0 REQUIRED)

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
