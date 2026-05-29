#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

"""Functional RBF-PUM interpolation test runner.

This script loads source/target point clouds, evaluates an analytic Franke
reference, calls ``pacman.rbf.interpolate``, and validates interpolation error
against mesh-dependent tolerances.
"""

import os
import sys
import numpy as np
import pacman
import argparse
from franke_functions import franke_3d


def test_interpolation(mesh_dir, mesh_file, method, execspace):
    """Run one RBF-PUM interpolation validation case.

    Parameters
    ----------
    mesh_dir : str
        Directory containing the ``.vtkhdf`` meshes used in the test pair.
    mesh_file : tuple[str, str]
        Pair ``(source_mesh, target_mesh)`` of mesh filenames.
    method : int
        RBF function constant from ``pacman.rbf.functions``.
        Supported values in this test script:
        - ``pacman.rbf.functions.WENDLANDC0``
        - ``pacman.rbf.functions.WENDLANDC2``
        - ``pacman.rbf.functions.WENDLANDC4``
        - ``pacman.rbf.functions.WENDLANDC6``
        - ``pacman.rbf.functions.WENDLANDC8``
    execspace : int
        Execution-space constant from ``pacman.execspaces``.
        Supported values in this test script:
        - ``pacman.execspaces.SERIAL``
        - ``pacman.execspaces.OPENMP``
        - ``pacman.execspaces.CUDA``
    """

    spaceDimension = 3
    source_mesh = np.load(mesh_dir + mesh_file[0])
    target_mesh = np.load(mesh_dir + mesh_file[1])

    tp = None
    sp_values = None

    sp = source_mesh["points"][:, :spaceDimension].astype(np.float64)
    tp = target_mesh["points"][:, :spaceDimension].astype(np.float64)

    sp_values = franke_3d(sp[:,0], sp[:,1], sp[:,2])

    tp_values = pacman.rbf.interpolate(spaceDimension, execspace, method, sp, sp_values, tp)

    tp_ref = franke_3d(tp[:,0], tp[:,1], tp[:,2])
    error_norm = np.linalg.norm(tp_values - tp_ref) / np.linalg.norm(tp_ref)
    max_error_allowed = 0.0
    if ('3' in mesh_file[0]):
        max_error_allowed = 1.5e-2
    if ('2' in mesh_file[0]):
        max_error_allowed = 1e-3
    if ('1' in mesh_file[0]):
        max_error_allowed = 1e-4
    if ('4' in mesh_file[0] or '5' in mesh_file[0]):
        max_error_allowed = 2e-5
    if (error_norm > max_error_allowed):
        print(f'Target Points interpolated values:\n{tp_values}\nTarget Points reference values:\n{tp_ref}')
        print(f'error norm: {error_norm} > {max_error_allowed}')
    assert error_norm < max_error_allowed, f"Method {method} for mesh_file {mesh_file}: error norm {error_norm} exceeds threshold"

def main():
    """Parse CLI arguments and execute one RBF-PUM interpolation test case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh")
    parser.add_argument("--method")
    parser.add_argument("--exec_space")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(script_dir, "./meshes/")

    methods = {
        "wendlandc0": pacman.rbf.functions.WENDLANDC0,
        "wendlandc2": pacman.rbf.functions.WENDLANDC2,
        "wendlandc4": pacman.rbf.functions.WENDLANDC4,
        "wendlandc6": pacman.rbf.functions.WENDLANDC6,
        "wendlandc8": pacman.rbf.functions.WENDLANDC8,
    }

    execspaces = {
        "SERIAL": pacman.execspaces.SERIAL,
        "OPENMP": pacman.execspaces.OPENMP,
        "CUDA": pacman.execspaces.CUDA,
        "HIP": pacman.execspaces.HIP
    }

    avail_execspaces = pacman.execspaces.available()

    requested_execspace = execspaces[args.exec_space]

    if args.exec_space not in avail_execspaces:
        print(
            f"[SKIP] Execution space {args.exec_space} not available "
            f"(available: {avail_execspaces})"
        )
        sys.exit(77)

    meshes= {
        # Coarse to Fine
        "coarse_to_fine_0": ("0.03.npz", "0.003.npz"),

        # Fine to Coarse
        "fine_to_coarse_0": ("0.003.npz", "0.03.npz"),

        # Same meshes
        "same_0": ("0.03.npz", "0.03.npz"),
    }

    test_interpolation(mesh_dir, meshes[args.mesh], methods[args.method], execspaces[args.exec_space])


if __name__ == "__main__":
    main()