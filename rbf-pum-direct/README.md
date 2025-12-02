# RBF-PUM Interpolator

## Get the source code

```shell
git clone https://git.cloud.safran/safransa/safrantech/mads/PACMAN.git && cd PACMAN/rbf-pum-direct
```

## Get the requirements

It is your responsibility to have these librairies properly installed in your environment:

Host-only interpolator requirements (Serial, Threads, OpenMP):

- `gcc@13+` (or any compiler with a full C++20 support)
- `cmake@3.31+`
- `kokkos@4.7+`
- `ArborX@2.0+`
- `Eigen@5.0+`

Additional device requirements (Cuda):

- `kokkoskernels@4.7+`
- `cuda@12.9+`

You can find an example of a spack env which is able to build and run this project in: `./env/spack.yaml`.

Note: some older versions of `cmake`, `Eigen` and `cuda` may work but have not been tested.

## Configure the CMake project

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -S . -B build
```

## Build the project

```shell
cmake --build build -- -j $(nproc)
```

## Using the Python bindings

Optional: add the build folder to `PYTHONPATH`:

```shell
export PYTHONPATH=${YOUR_BUILD_FOLDER}:$PYTHONPATH
```

You can import and use the python module from any python file:

```py
import RbfPumInterpolator

out = RbfPumInterpolator(..., ..., ...)
```
