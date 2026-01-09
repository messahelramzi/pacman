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
# Build the Python Module if required
# -----------------------------------------------------
if (BUILD_MODULE)
    add_subdirectory(pybindings)
endif()

# -----------------------------------------------------
# Export targets
# -----------------------------------------------------
install(TARGETS rbfpum finiteelements PACMAN
        EXPORT PACMANTargets
)

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