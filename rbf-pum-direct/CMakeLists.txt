cmake_minimum_required(VERSION 3.31)

project(MyArborX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(ArborX REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  ArborX::ArborX
)

target_compile_features(main PRIVATE cxx_std_17)

target_compile_definitions(main PRIVATE KOKKOS_ENABLE_CUDA)

target_compile_options(main PRIVATE
  -O3
  -g0
)
