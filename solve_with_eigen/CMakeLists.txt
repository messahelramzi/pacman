cmake_minimum_required(VERSION 3.31)

project(SolveWithEigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE
    Eigen3::Eigen
)

target_compile_features(main PRIVATE cxx_std_17)

target_compile_options(main PRIVATE
    -m64
)
