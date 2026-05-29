#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

add_library(finiteelements INTERFACE)

target_include_directories(finiteelements INTERFACE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(finiteelements INTERFACE
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
)
target_compile_features(finiteelements INTERFACE cxx_std_20)
