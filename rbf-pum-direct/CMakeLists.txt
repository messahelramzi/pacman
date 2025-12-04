cmake_minimum_required(VERSION 3.31)

project(RBFPUMDirect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

# Required only for main.cpp
find_package(VTK COMPONENTS
    CommonCore
    IOXML
    CommonDataModel
    REQUIRED)
# Required for the Host Solver (Serial/Threads/OpenMP backends) and Device Solver (Cuda backend)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/external/installs/kokkos-5.0.0/lib64/cmake/Kokkos CACHE PATH "")
set(ArborX_DIR ${CMAKE_SOURCE_DIR}/external/installs/ArborX-2.0.1/share/cmake/ArborX CACHE PATH "")
set(KokkosKernels_DIR ${CMAKE_SOURCE_DIR}/external/installs/KokkosKernels-5.0.0/lib64/cmake/KokkosKernels CACHE PATH "")
find_package(Kokkos 4.7 REQUIRED)
find_package(ArborX 2.0 REQUIRED)
find_package(KokkosKernels 4.7 REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
    Eigen3::Eigen
    VTK::CommonCore
    VTK::IOXML
    VTK::CommonDataModel
)
target_include_directories(main PRIVATE "src/")

target_compile_features(main PRIVATE cxx_std_20)
target_compile_definitions(main PRIVATE Kokkos_ENABLE_CUDA)
target_compile_options(main PRIVATE
    -m64
)

## Python Bindings (Host Solver Only)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(RbfPumInterpolator "src/pybindings/bindings.cpp")
target_link_libraries(RbfPumInterpolator PRIVATE
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
    Eigen3::Eigen
)
target_include_directories(RbfPumInterpolator PRIVATE "src/")
