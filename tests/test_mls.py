#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

"""Functional MLS interpolation test — synthetic grids.

Tests MLS interpolation from a coarse 5^D grid to a fine 10^D grid
in 1-D, 2-D and 3-D using analytic Franke reference functions.
"""

import sys
import numpy as np
import pacman
from franke_functions import franke_1d, franke_2d, franke_3d


def _make_grid(n_per_dim: int, dim: int) -> np.ndarray:
    """Return a regular grid of ``n_per_dim^dim`` points in [0,1]^dim.

    Returns
    -------
    numpy.ndarray, shape (n_per_dim**dim, dim), dtype float64
    """
    axes = [np.linspace(0.0, 1.0, n_per_dim)] * dim
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    return pts.astype(np.float64)


def _reference(pts: np.ndarray) -> np.ndarray:
    dim = pts.shape[1]
    if dim == 1:
        return franke_1d(pts[:, 0])
    if dim == 2:
        return franke_2d(pts[:, 0], pts[:, 1])
    return franke_3d(pts[:, 0], pts[:, 1], pts[:, 2])


def test_mls_grid(dim: int, n_src: int, n_tgt: int, execspace: int,
                  tol: float) -> None:
    """Interpolate from an ``n_src^dim`` source grid to an ``n_tgt^dim``
    target grid and assert the relative L2 error is below *tol*.
    """
    sp = _make_grid(n_src, dim)
    tp = _make_grid(n_tgt, dim)

    sp_values = _reference(sp)
    tp_ref    = _reference(tp)

    tp_values = pacman.MLS.interpolate(dim, execspace, sp, sp_values, tp)

    error_norm = np.linalg.norm(tp_values - tp_ref) / np.linalg.norm(tp_ref)
    assert error_norm < tol, (
        f"MLS dim={dim} src={n_src}^{dim} tgt={n_tgt}^{dim}: "
        f"rel_l2={error_norm:.3e} > tol={tol:.3e}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exec_space", default="SERIAL")
    args = parser.parse_args()

    execspaces = {
        "SERIAL": pacman.execspaces.SERIAL,
        "OPENMP": pacman.execspaces.OPENMP,
        "CUDA":   pacman.execspaces.CUDA,
        "HIP":    pacman.execspaces.HIP,
    }

    avail_execspaces = pacman.execspaces.available()
    if args.exec_space not in avail_execspaces:
        print(
            f"[SKIP] Execution space {args.exec_space} not available "
            f"(available: {avail_execspaces})"
        )
        sys.exit(77)

    execspace = execspaces[args.exec_space]

    # (dim, n_src, n_tgt, tol)
    cases = [
        (1,  20, 20, 1e-4),
        (2,  20, 20, 1e-4),
        (3,  20, 20, 1e-4),
    ]

    for dim, n_src, n_tgt, tol in cases:
        test_mls_grid(dim, n_src, n_tgt, execspace, tol)
        print(f"  PASS  dim={dim}  src={n_src}^{dim}->tgt={n_tgt}^{dim}")


if __name__ == "__main__":
    main()
