cmake_minimum_required(VERSION 3.31)

project(RBFPUMDirect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

# Required only for main
find_package(VTK COMPONENTS
    CommonCore
    IOXML
    CommonDataModel
    REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# find_package(OpenMP REQUIRED)
# find_package(hwloc REQUIRED)

# set(GINKGO_LIBS_DIR "/data/ssa/users/d647795/ginkgo-1.10.0/build/lib")
# set(GINKGO_INCLUDE_DIR "/data/ssa/users/d647795/ginkgo-1.10.0/build/install/include")
# add_library(Ginkgo::ginkgo SHARED IMPORTED)
# set_target_properties(Ginkgo::ginkgo PROPERTIES
#     IMPORTED_LOCATION "${GINKGO_LIBS_DIR}/libginkgo.so"
#     INTERFACE_INCLUDE_DIRECTORIES "${GINKGO_INCLUDE_DIR}"
# )

# set(KOKKOS_LIBS_DIR "/data/ssa/users/d647795/kokkos-5.0.0/install/lib64")
# set(KOKKOS_INCLUDE_DIR "/data/ssa/users/d647795/kokkos-5.0.0/install/include")
# add_library(Kokkos5 STATIC IMPORTED GLOBAL)
# set_target_properties(Kokkos5 PROPERTIES
#     IMPORTED_LOCATION "${KOKKOS_LIBS_DIR}/libkokkoscore.a"
#     INTERFACE_LINK_LIBRARIES "${KOKKOS_LIBS_DIR}/libkokkosalgorithms.a;${KOKKOS_LIBS_DIR}/libkokkoscontainers.a;${KOKKOS_LIBS_DIR}/libkokkossimd.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${KOKKOS_INCLUDE_DIR}"
# )

# set(KOKKOS_KERNELS_LIBS_DIR "/data/ssa/users/d647795/kokkos-kernels-develop/install/lib64")
# set(KOKKOS_KERNELS_INCLUDE_DIR "/data/ssa/users/d647795/kokkos-kernels-develop/install/include")
# add_library(KokkosKernels5 STATIC IMPORTED)
# set_target_properties(KokkosKernels5 PROPERTIES
#     IMPORTED_LOCATION "${KOKKOS_KERNELS_LIBS_DIR}/libkokkoskernels.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${KOKKOS_KERNELS_INCLUDE_DIR}"
# )

# set(ARBORX_INCLUDE_DIR "/data/ssa/users/d647795/ArborX-2.0.1/install/include")
# add_library(ArborX2 INTERFACE)
# target_include_directories(ArborX2 INTERFACE
#     "${ARBORX_INCLUDE_DIR}"
#     "${ARBORX_INCLUDE_DIR}/ArborX"
#     "${ARBORX_INCLUDE_DIR}/ArborX/cluster"
#     "${ARBORX_INCLUDE_DIR}/ArborX/cluster/detail"
#     "${ARBORX_INCLUDE_DIR}/ArborX/distributed"
#     "${ARBORX_INCLUDE_DIR}/ArborX/distributed/detail"
#     "${ARBORX_INCLUDE_DIR}/ArborX/geometry"
#     "${ARBORX_INCLUDE_DIR}/ArborX/geometry/algorithms"
#     "${ARBORX_INCLUDE_DIR}/ArborX/interpolation"
#     "${ARBORX_INCLUDE_DIR}/ArborX/interpolation/detail"
#     "${ARBORX_INCLUDE_DIR}/ArborX/kokkos_ext"
#     "${ARBORX_INCLUDE_DIR}/ArborX/misc"
#     "${ARBORX_INCLUDE_DIR}/ArborX/spatial"
#     "${ARBORX_INCLUDE_DIR}/ArborX/spatial/detail"
# )
# target_link_libraries(ArborX2 INTERFACE Kokkos5)

# Required for the Host Solver (Serial/Threads/OpenMP backends)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/external/installs/kokkos-5.0.0/lib64/cmake/Kokkos CACHE PATH "")
set(ArborX_DIR ${CMAKE_SOURCE_DIR}/external/installs/ArborX-2.0.1/share/cmake/ArborX CACHE PATH "")
set(KokkosKernels_DIR ${CMAKE_SOURCE_DIR}/external/installs/KokkosKernels-5.0.0/lib64/cmake/KokkosKernels CACHE PATH "")
find_package(Kokkos 5.0 REQUIRED)
find_package(ArborX 2.0 REQUIRED)
find_package(KokkosKernels 5.0 REQUIRED)

# Required for the Host Solver (Serial/Threads/OpenMP backends)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/external/installs/kokkos-5.0.0/lib64/cmake/Kokkos CACHE PATH "")
set(ArborX_DIR ${CMAKE_SOURCE_DIR}/external/installs/ArborX-2.0.1/share/cmake/ArborX CACHE PATH "")
set(KokkosKernels_DIR ${CMAKE_SOURCE_DIR}/external/installs/KokkosKernels-5.0.0/lib64/cmake/KokkosKernels CACHE PATH "")
find_package(Kokkos 5.0 REQUIRED)
find_package(ArborX 2.0 REQUIRED)
find_package(KokkosKernels 5.0 REQUIRED)

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

target_compile_definitions(main PRIVATE KOKKOS_ENABLE_CUDA)

# set(CMAKE_CXX_COMPILER "/data/ssa/users/d647795/kokkos-5.0.0/bin/nvcc_wrapper")

target_compile_options(main PRIVATE
    -m64
    # -Wall
    # -Wextra
    # -Wuseless-cast
    # -Wuninitialized
    # -Wmaybe-uninitialized
    # -Wstrict-aliasing
    # -Wsuggest-attribute=pure
    # -Wsuggest-attribute=const
    # -Wsuggest-attribute=noreturn
    # -Wmissing-noreturn
    # -Warith-conversion
    # -Wfloat-conversion
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
