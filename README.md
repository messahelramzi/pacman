# PACMAN (Parallel Application Coupling for Multiphysics And Numerical simulation)

## Get the source code

```shell
git clone https://git.cloud.safran/safransa/safrantech/mads/PACMAN.git && cd PACMAN/
```

## Get the requirements

It is your responsibility to have these librairies properly installed in your environment:

Host-only interpolator requirements (Serial, Threads, OpenMP):

- `gcc@13+` (or any compiler with a full C++20 support)
- `cmake@3.31+`
- `kokkos@4.7+`
- `kokkoskernels@4.7+`
- `ArborX@2.0+`

Additional device requirements (Cuda):

- `cuda@12.9+`

Additional requirements for the Python module:

- `pybind11@2.13+`

You can find an example of a spack env which is able to build and run this project in: `./env/spack.yaml`.

Note: some older versions of these dependencies may work but have not been tested.

## Configure the CMake project

List of available custom options:

- `BUILD_MODULE`: BOOL = Enable the build of the Python bindings. Requires `pybind11`. Default value: `ON`.
- `BUILD_TESTS`: BOOL = Enable the build of the test binary. Requires `vtk`. Default value: `OFF`.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MODULE=ON -DBUILD_TESTS=OFF -G Ninja -S . -B build
```

The provided build system overrides some CMake options, which should not be modified by the user:

- `CMAKE_POSITION_INDEPENDENT_CODE`, set to `ON` to allow the Python module to be linked with device librairies.
- `CMAKE_INTERPROCEDURAL_OPTIMIZATION`, set to `OFF` to allow compilation with CUDA.

The PACMAN library exports one CMake target, which is an interface called `PACMAN::PACMAN`. To use PACMAN in your project, you can use `find_package`:

```cmake
find_package(PACMAN 0.1.0 REQUIRED)
target_link_libraries(your_target PRIVATE PACMAN::PACMAN)
```

## Build the project

```shell
cmake --build build -- -j $(nproc)
```

## Install the project

```shell
cmake --build build --target install -- -j $(nproc)
```

## Use the Python bindings

Optional: Add the install folder to `PYTHONPATH` if it is not already here:

```shell
export PYTHONPATH=${YOUR_INSTALL_FOLDER}:$PYTHONPATH
```

You can import and use the python module from any python file:

```py
import pacman
```

If you encounter this kind of error when importing the Python module:

```shell
ImportError: libkokkoskernels.so: cannot open shared object file: No such file or directory
```

Make sure that the dependencies' dynamic librairies like `libcuda.so`, `libcudart.so`, `libkokkoscore.so` or `libkokkoskernels.so` are visible from `LD_LIBRARY_PATH`. Maybe these librairies are uncorrectly installed.

## The Python Module

The available functions in the module are:

```py
pacman.fe.interpolate(execspace, method, source_points, source_values, conn_val, conn_off, cell_types, target_points)
pacman.fe.interpolate(execspace, method, source_points, source_values, target_points)
```

`execspace`: `char`, constant value defined in the `pacman` module to target a given backend.  
`method`: `char`, constant value defined in the `pacman.fe.methods` submodule to use a given finite elements method.  
`source_points`: `numpy.array`, a 2D `np.array` which contains the points of the source mesh. It must be shaped like `(n, 2)` for a 2D mesh or `(n, 3)` for a 3D mesh.  
`source_values`: `numpy.array`, a 1D `np.array` which contains the data associated to each point. The values must follow the order given in `source_points`.  
`conn_val`: `numpy.array`, TODO  
`conn_off`: `numpy.array`, TODO  
`cell_types`: `numpy.array`, TODO  
`target_points`: `numpy.array`, a 2D `np.array` which contains the points of the target mesh. It must be shaped like `(m, 2)` for a 2D mesh or `(m, 3)` for a 3D mesh.  
Returns: `numpy.array`, a 1D `np.array` which contains the interpolated data at the target points coordinates using the given method.

This function interpolates points data from `source_points` to `target_points`. It uses the `execspace` argument to define the execution space of the function, and `method` to define the finite elements method to use. Please see below for the available execution spaces and FE methods.

The connectivity is not used in the case of Nearest/Nearest method. Thus, there is a prototype without connectivity data or cell types.

```py
pacman.rbf.interpolate(execspace, rbf_function, source_points, source_values, target_points)
```

`execspace`: `char`, constant value defined in the `pacman` module to target a given backend.  
`rbf_function`: `char`, constant value defined in the `pacman.rbf.functions` submodule to use a given RBF function.  
`source_points`: `numpy.array`, a 2D `np.array` which contains the points of the source mesh. It must be shaped like `(n, 2)` for a 2D mesh or `(n, 3)` for a 3D mesh.  
`source_values`: `numpy.array`, a 1D `np.array` which contains the data associated to each point. The values must follow the order given in `source_points`.  
`target_points`: `numpy.array`, a 2D `np.array` which contains the points of the target mesh. It must be shaped like `(m, 2)` for a 2D mesh or `(m, 3)` for a 3D mesh.  
Returns: `numpy.array`, a 1D `np.array` which contains the interpolated data at the target points coordinates using the given method.

This function interpolates points data from `source_points` to `target_points`. It uses the `execspace` argument to define the execution space of the function, and `rbf_function` to define the RBF function to use. Please see below for the available execution spaces and RBF functions.

This function works with unstructured data cloud and does not require connectivity data.

## Defined constants for function calls

There are constants defined as submodules to pass the execution space or the interpolation method to the function call. These are typed as char/unsigned char, but you must not rely on their underlying raw value, and use the module defined constants.

Execution spaces: (naming follows Kokkos execution spaces, ticked is tested)

- [x] `pacman.execspaces.SERIAL`
- [x] `pacman.execspaces.OPENMP`
- [ ] `pacman.execspaces.THREADS`
- [x] `pacman.execspaces.CUDA`
- [ ] `pacman.execspaces.HIP`
- [ ] `pacman.execspaces.SYCL`

RBF functions for the RBF-PUM interpolation method (ticked is tested):

- [x] `pacman.rbf.functions.WENDLANDC0`
- [x] `pacman.rbf.functions.WENDLANDC2`
- [x] `pacman.rbf.functions.WENDLANDC4`
- [x] `pacman.rbf.functions.WENDLANDC6`
- [x] `pacman.rbf.functions.WENDLANDC8`

Finite elements methods (ticked is working):

- [x] `pacman.fe.methods.NEAREST_NEAREST`
- [ ] `pacman.fe.methods.INTERP_CLAMP`
- [ ] `pacman.fe.methods.INTERP_NEAREST`
- [ ] `pacman.fe.methods.INTERP_ZEROFILL`
- [ ] `pacman.fe.methods.INTERP_EXTRAP`

## Additional notes about the use of Kokkos

The `pacman` module automatically initialize Kokkos when imported, and finalize Kokkos at exit, but still provides `pacman.initialize()` and `pacman.finalize()` to manage Kokkos manually. This should not be used, the module can manage Kokkos by itself.

## Project notes

The project is structured as follows:

- `cmake/`: CMake config files to build the package.
- `env/`: Spack dev environment file, not mandatory to use.
- `src/`: source files for the PACMAN library.
- `tests/`: Unit tests if necessary and a fonctional test executable.

The source files are structured as follows:

- `common/`: shared headers across the whole library, every resource used by multiple interpolation methods must be in this folder.
- `finite_elements/`: headers that provides the private interface of the finite elements interpolation methods.
- `pybindings/`: source files and Python bindings headers. The module public interface must be included in `bindings.cpp`, and this is the only thing this file should contain.
- `rbf_pum/`: headers that provides the private interface of the RBF-PUM interpolation method.
- `interpolate.hpp`: the only public interface header.

The PACMAN library is headers only. The `.hpp` extension should be used for all of the header files which contain functions. The `.hxx` extension should be use for headers which contains data structures with no function or inlined functions.

The naming of the variables, functions and files must describe what it is meant to do. We use `PascalCase` for namespaces, class names and functions names. We use `camelCase` for class attributes and function arguments. We use `snake_case` for the stack variables. Following these rules is not mandatory but appreciated.

The whole PACMAN code should be in a namespace named `PACMAN`. Each subpart, each interpolation system should be in its own inner namespace, named accordingly to the interpolation system. Even if the Python bindings are semantically separated from the library code, they should live in the namespace `PACMAN` and should live in their own inner namespaces.  
The only part of the code which is allowed to be outside of the namespace `PACMAN` is the `ArborX::AccessTraits` specializations for custom predicates, which must be in a top-level `ArborX` namespace, for convenience and readability.

Any new interpolation system, which is not a finite elements methods, must be added in its own directory inside of `src/`, and should export its own interface target, create its entry point in the Python bindings if required. For simplicity and performances, the entry point of every interpolation method should be a reference to a `Transfer` object.

The Python module binding requires at least one source file, with the extension `.cpp`. It is a good practice to use multiple source files, one per interpolation method family, to reduce the compilation time. However, the declaration of the Python module should remain in `bindings.cpp` for clarity.

The Python bindings are meant to be as fast as possible. The given structure allow to call the C++ underlying interpolation functions with one copy of the data only (the argument conversion performed by `pybind11`). We assume that the C style numpy array created by `pybind11` is always contiguous. Also, we use `std::variant` and `std::visit` to generate template specializations according to the compile flags passed to Kokkos. This system increases the compilation time but reduces the runtime overhead of the Python interface.
