set(PYTHONPATH ${CMAKE_BINARY_DIR} $ENV{PYTHONPATH})

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME python_test_dummy
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/dummy.py
)
set_tests_properties(python_test_dummy PROPERTIES
  ENVIRONMENT
    "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
)
include(PACMANAddFETests)

set(MESH_FILES aste hexahedron pyramid quad tetra triangle wedge)
set(METHODS nearest_nearest interp_clamp interp_nearest interp_zerofill interp_extrap)
set(EXECSPACES serial openmp cuda)

foreach(exec_space IN LISTS EXECSPACES)
  foreach(mesh IN LISTS MESH_FILES)

    if(mesh STREQUAL "aste")
      add_fe_test(${mesh} "interp_clamp" ${exec_space})
    else()
      foreach(method IN LISTS METHODS)
        add_fe_test(${mesh} ${method} ${exec_space})
      endforeach()
    endif()

  endforeach()
endforeach()
