set(PYTHONPATH ${CMAKE_BINARY_DIR} $ENV{PYTHONPATH})

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Include Python module test

add_test(
    NAME python_test_dummy
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/dummy.py
)
set_tests_properties(python_test_dummy PROPERTIES
  ENVIRONMENT
    "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
)

# Tests

set(EXECSPACES serial openmp cuda)

# Tests FE
include(PACMANAddFETests)
set(FE_METHODS nearest_nearest interp_clamp interp_nearest interp_zerofill interp_extrap)
set(FE_MESH_FILES aste hexahedron pyramid quad tetra triangle wedge)

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
set(RBF_MESHES coarse_to_fine_0 coarse_to_fine_1 coarse_to_fine_2 coarse_to_fine_3
               fine_to_coarse_0 fine_to_coarse_1 fine_to_coarse_2 fine_to_coarse_3
               same_0 same_1 same_2)
set(RBF_FUNCTIONS wendlandc0 wendlandc2 wendlandc4 wendlandc6 wendlandc8)

foreach(exec_space IN LISTS EXECSPACES)
  foreach(mesh IN LISTS RBF_MESHES)
      foreach(method IN LISTS RBF_FUNCTIONS)
        add_rbf_test(${mesh} ${method} ${exec_space})
      endforeach()
  endforeach()
endforeach()
