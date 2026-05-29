#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

set(PYTHONPATH ${CMAKE_BINARY_DIR} $ENV{PYTHONPATH})

find_package(Python3 REQUIRED COMPONENTS Interpreter)

#==== Test include Python module pacman
add_test(
    NAME python_test_import
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/test_import.py
)
set_tests_properties(python_test_import PROPERTIES
  ENVIRONMENT
    "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
)

#==== Test pybindings
add_test(
    NAME python_test_pybindings
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/test_pybindings.py
)
set_tests_properties(python_test_pybindings PROPERTIES
  ENVIRONMENT
    "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
)

#==== C++ interface test (equivalent of test_pybindings.py)
add_executable(test_cpp_fe_interface ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_fe_interface.cpp)
target_link_libraries(test_cpp_fe_interface PRIVATE pacman_interface_cpp)
add_test(NAME cpp_test_fe_interface COMMAND test_cpp_fe_interface)

#==== C++ RBF interface test (equivalent of test_rbf-pum.py — OpenMP only)
add_executable(test_cpp_rbf_interface ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_rbf_interface.cpp)
target_link_libraries(test_cpp_rbf_interface PRIVATE pacman_interface_cpp)
add_test(NAME cpp_test_rbf_interface COMMAND test_cpp_rbf_interface)

#===== Tests FE/RBF interpolation methods and execution spaces
set(EXECSPACES SERIAL OPENMP CUDA HIP)

# Tests FE
include(PACMANAddFETests)
set(FE_METHODS nearest_nearest interp_clamp interp_nearest interp_zerofill interp_extrap)
set(FE_MESH_FILES triangle triangle6 quad quad8 tetra tetra10 pyramid pyramid13 wedge wedge15 hexahedron hexahedron20 aste)

foreach(exec_space IN LISTS EXECSPACES)
  foreach(mesh IN LISTS FE_MESH_FILES)
    if(mesh STREQUAL "aste")
      add_fe_test(${mesh} "interp_clamp" ${exec_space})
    else()
      foreach(method IN LISTS FE_METHODS)
        add_fe_test(${mesh} ${method} ${exec_space})
      endforeach()
    endif()
  endforeach()
endforeach()

# Tests RBF-PUM
include(PACMANAddRBFTests)
set(RBF_MESHES coarse_to_fine_0 fine_to_coarse_0 same_0)
set(RBF_FUNCTIONS wendlandc0 wendlandc2 wendlandc4 wendlandc6 wendlandc8)

# Tests RBF-PUM
foreach(exec_space IN LISTS EXECSPACES)
  foreach(mesh IN LISTS RBF_MESHES)
      foreach(method IN LISTS RBF_FUNCTIONS)
        add_rbf_test(${mesh} ${method} ${exec_space})
      endforeach()
  endforeach()
endforeach()

#==== Fortran interface tests
if (BUILD_FORTRAN_INTERFACE)
    add_executable(test_fortran_rbf_interface
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fortran_rbf_interface.f90)
    target_link_libraries(test_fortran_rbf_interface PRIVATE pacman_interface_fortran)
    add_test(NAME fortran_test_rbf_interface COMMAND test_fortran_rbf_interface)

    add_executable(test_fortran_fe_interface
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fortran_fe_interface.f90)
    target_link_libraries(test_fortran_fe_interface PRIVATE pacman_interface_fortran)
    add_test(NAME fortran_test_fe_interface COMMAND test_fortran_fe_interface)
endif()

# Add skip test property when test returns exit code 77
get_property(ALL_TESTS DIRECTORY PROPERTY TESTS)

foreach(test_name IN LISTS ALL_TESTS)
  set_tests_properties(${test_name}
    PROPERTIES
      SKIP_RETURN_CODE 77
  )
endforeach()