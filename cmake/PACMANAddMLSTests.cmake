#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

function(add_mls_test exec_space)
  add_test(
    NAME mls_${exec_space}
    COMMAND python3
            ${CMAKE_CURRENT_SOURCE_DIR}/test_mls.py
            --exec_space ${exec_space}
  )
  set_tests_properties(mls_${exec_space} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PACMAN_PYTHON_DIR}:$ENV{PYTHONPATH}"
  )
endfunction()
