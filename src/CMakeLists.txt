#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

# -----------------------------------------------------
# Build all of the targets
# -----------------------------------------------------
add_subdirectory(rbf_pum)
add_subdirectory(finite_elements)

# -----------------------------------------------------
# Create the public target which will be exported
# -----------------------------------------------------
add_library(PACMAN INTERFACE)
add_library(PACMAN::PACMAN ALIAS PACMAN)
target_link_libraries(PACMAN INTERFACE
    rbfpum
    finiteelements
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
)
target_include_directories(PACMAN INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(PACMAN INTERFACE cxx_std_20)

# -----------------------------------------------------
# C++ interface library (no Python dependency)
# Mirrors the Python pybindings API using raw pointers.
# Library type follows BUILD_SHARED_LIBS (same default as rbfpum/finiteelements).
# -----------------------------------------------------
add_library(pacman_interface_cpp
    interface.cpp
    interface_eti_serial.cpp
    interface_eti_openmp.cpp
    interface_eti_threads.cpp
    interface_eti_cuda.cpp
    interface_eti_hip.cpp
    interface_eti_sycl.cpp
    rbf_interface.cpp
    rbf_interface_eti_serial.cpp
    rbf_interface_eti_openmp.cpp
    rbf_interface_eti_threads.cpp
    rbf_interface_eti_cuda.cpp
    rbf_interface_eti_hip.cpp
    rbf_interface_eti_sycl.cpp
)
add_library(PACMAN::pacman_interface_cpp ALIAS pacman_interface_cpp)
target_link_libraries(pacman_interface_cpp PUBLIC PACMAN)
target_include_directories(pacman_interface_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/PACMAN>
)
set_target_properties(pacman_interface_cpp PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# -----------------------------------------------------
# Fortran module library (optional — requires BUILD_FORTRAN_INTERFACE)
# Compiles pacman_fortran.f90 into a Fortran module and links to
# pacman so Fortran callers only need one target.
# -----------------------------------------------------
if (BUILD_FORTRAN_INTERFACE)
    add_library(pacman_interface_fortran
        fortran_interface.cpp
        pacman_fortran.f90
    )
    add_library(PACMAN::pacman_interface_fortran ALIAS pacman_interface_fortran)
    target_link_libraries(pacman_interface_fortran PUBLIC pacman_interface_cpp)
    set_target_properties(pacman_interface_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules
    )
    target_include_directories(pacman_interface_fortran PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>
        $<INSTALL_INTERFACE:include/PACMAN/fortran>
    )
    set_target_properties(pacman_interface_fortran PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# -----------------------------------------------------
# Build the Python Module if required
# -----------------------------------------------------
if (BUILD_MODULE)
    add_subdirectory(pybindings)
endif()

# -----------------------------------------------------
# Export targets
# -----------------------------------------------------
install(TARGETS rbfpum finiteelements PACMAN pacman_interface_cpp
        EXPORT PACMANTargets
)

if (BUILD_FORTRAN_INTERFACE)
    install(TARGETS pacman_interface_fortran
            EXPORT PACMANTargets
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/
            DESTINATION include/PACMAN/fortran
            FILES_MATCHING PATTERN "*.mod"
    )
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
        DESTINATION include/PACMAN
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
)

install(EXPORT PACMANTargets
    NAMESPACE PACMAN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PACMAN
)