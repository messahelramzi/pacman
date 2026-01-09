find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pacman bindings.cpp fe_bindings.cpp rbf_bindings.cpp)
target_link_libraries(pacman PRIVATE PACMAN)
set_target_properties(pacman PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# -----------------------------------------------------
# Install Python module
# -----------------------------------------------------
install(TARGETS pacman
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
