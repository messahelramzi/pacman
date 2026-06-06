#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pacman
    bindings.cpp
    fe_bindings.cpp
    rbf_bindings.cpp
    # RBF explicit template instantiation — one TU per exec space so they
    # compile in parallel and each carries only 15 specializations (3 dims ×
    # 5 RBF functions) rather than the full 90-combo cross-product.
    rbf_eti_serial.cpp
    rbf_eti_openmp.cpp
    rbf_eti_threads.cpp
    rbf_eti_cuda.cpp
    rbf_eti_hip.cpp
    rbf_eti_sycl.cpp
    # FE explicit template instantiation — one TU per exec space (3 dims each).
    fe_eti_serial.cpp
    fe_eti_openmp.cpp
    fe_eti_threads.cpp
    fe_eti_cuda.cpp
    fe_eti_hip.cpp
    fe_eti_sycl.cpp
    # MLS explicit template instantiation — one TU per exec space (3 dims each).
    mls_bindings.cpp
    mls_eti_serial.cpp
    mls_eti_openmp.cpp
    mls_eti_threads.cpp
    mls_eti_cuda.cpp
    mls_eti_hip.cpp
    mls_eti_sycl.cpp
)
target_link_libraries(pacman PRIVATE PACMAN)
set_target_properties(pacman PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set(PACMAN_PYTHON_DIR
    $<TARGET_FILE_DIR:pacman> CACHE PATH ".")

if (ENABLE_COVERAGE)
    target_compile_options(pacman PRIVATE -g -O0 --coverage)
    target_link_options(pacman PRIVATE --coverage)
endif()

# -----------------------------------------------------
# Install Python module
# -----------------------------------------------------
install(TARGETS pacman
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
