cmake_minimum_required(VERSION 3.31)

project(MyArborX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(ArborX REQUIRED)
find_package(VTK COMPONENTS
    vtkCommonCore
    vtkIOXML
    vtkCommonDataModel
    REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
    ArborX::ArborX
    VTK::CommonCore
    VTK::IOXML
    VTK::CommonDataModel
)

target_compile_features(main PRIVATE cxx_std_17)

target_compile_definitions(main PRIVATE KOKKOS_ENABLE_CUDA)

target_compile_options(main PRIVATE
    -O0
    -g3
)
