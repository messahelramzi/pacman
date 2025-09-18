cmake_minimum_required(VERSION 3.20)
project(medcoupling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main interpolation_evaluation.cpp)

file(GLOB MED_FILES "${CMAKE_SOURCE_DIR}/*.med")
file(COPY ${MED_FILES} DESTINATION ${CMAKE_BINARY_DIR})

target_include_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/hdf5-1.10.11-mbrfezbmh4jx7ipuvkx4woo65wca7m5l/include")
target_link_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/hdf5-1.10.11-mbrfezbmh4jx7ipuvkx4woo65wca7m5l/lib")

target_include_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/salome-medcoupling-9.13.0-jfdd6tjxxjtubd77pgoovnypxbhpj27l/include")
target_link_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/salome-medcoupling-9.13.0-jfdd6tjxxjtubd77pgoovnypxbhpj27l/lib")

target_include_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/openmpi-5.0.8-mq5ma7ympjqcare25qnd2cyozt4egdba/include")
target_link_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/openmpi-5.0.8-mq5ma7ympjqcare25qnd2cyozt4egdba/lib")

target_include_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/salome-med-4.1.1-rvhz7mr5cynxdya6nxaq4j4ed4dxwjb6/include/")
target_link_directories(main PUBLIC "/gpfs/apps/ssa/spack/spack-v1.0.1/opt/spack/linux-zen3/salome-med-4.1.1-rvhz7mr5cynxdya6nxaq4j4ed4dxwjb6/lib/")

target_include_directories(main PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(main PUBLIC ${HDF5_LIBRARIES})

target_link_libraries(main PUBLIC medcoupling)
target_link_libraries(main PUBLIC medcouplingremapper)
target_link_libraries(main PUBLIC interpkernel)
target_link_libraries(main PUBLIC mpi)
target_link_libraries(main PUBLIC medloader)
