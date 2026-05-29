#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

function(add_rbf_test mesh method exec_space)
  string(REPLACE "/" "_" method_safe "${method}")

  add_test(
    NAME interp_${mesh}_${method_safe}_${exec_space}
    COMMAND python3
            ${CMAKE_CURRENT_SOURCE_DIR}/test_rbf-pum.py
            --mesh ${mesh}
            --method ${method}
            --exec_space ${exec_space}
  )
  set_tests_properties(interp_${mesh}_${method_safe}_${exec_space} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
  )
endfunction()
