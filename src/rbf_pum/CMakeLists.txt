add_library(rbfpum INTERFACE)

target_include_directories(rbfpum INTERFACE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(rbfpum INTERFACE
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
)
target_compile_features(rbfpum INTERFACE cxx_std_20)
