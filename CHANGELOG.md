# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - YYYY-MM-DD

### Added
- Initial public release of PACMAN as a header-only C++ library for interpolation transfer workflows.
- Public CMake package target `PACMAN::PACMAN` and install/export support.
- Core interpolation entry point in `src/interpolate.hpp` with shared common abstractions.
- Finite-elements interpolation methods:
	- `NEAREST_NEAREST`
	- `INTERP_CLAMP`
	- `INTERP_NEAREST`
	- `INTERP_ZEROFILL`
	- `INTERP_EXTRAP`
- RBF-PUM interpolation implementation with Wendland kernels (`C0`, `C2`, `C4`, `C6`, `C8`).
- Python bindings via `pybind11` for FE and RBF interpolation APIs.
- Execution-space constants for backend selection (Serial, OpenMP, CUDA, plus optional backends).
- CMake test integration and Python functional tests, including mesh-based interpolation datasets in `tests/meshes/`.

### Notes
- This is the first tagged release series (`v0.1.x`) and establishes the initial public API baseline.
