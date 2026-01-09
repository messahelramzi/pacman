find_package(VTK REQUIRED COMPONENTS
    CommonCore
    IOXML
    CommonDataModel
)

add_executable(test tests.cpp)

target_link_libraries(test PRIVATE
    PACMAN::PACMAN
    VTK::CommonCore
    VTK::IOXML
    VTK::CommonDataModel
)

target_compile_features(test PRIVATE cxx_std_20)

target_compile_options(test PRIVATE
    -m64
    -lineinfo
)