#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

"""Functional FE interpolation test runner.

This script loads source/target meshes, evaluates analytic Franke references,
calls ``pacman.fe.interpolate``, and validates results for multiple FE methods.
"""

import os
import sys
import pacman
import numpy as np
import argparse
from franke_functions import franke_1d, franke_2d, franke_3d


def test_interpolation(mesh_dir, mesh_file, method, execspace):
    """Run one FE interpolation validation case.

    Parameters
    ----------
    mesh_dir : str
        Directory containing the ``*_source.vtk`` and ``*_target.vtk`` meshes.
    mesh_file : str
        Base mesh name (without ``_source.vtk`` / ``_target.vtk`` suffix).
    method : int
        Finite-elements interpolation method constant from ``pacman.fe.methods``.
        Supported values in this test script:
        - ``pacman.fe.methods.NEAREST_NEAREST``
        - ``pacman.fe.methods.INTERP_CLAMP``
        - ``pacman.fe.methods.INTERP_NEAREST``
        - ``pacman.fe.methods.INTERP_ZEROFILL``
        - ``pacman.fe.methods.INTERP_EXTRAP``
    execspace : int
        Execution-space constant from ``pacman.execspaces``.
        Supported values in this test script (tested so far):
        - ``pacman.execspaces.SERIAL``
        - ``pacman.execspaces.OPENMP``
        - ``pacman.execspaces.CUDA``
    """

    source_mesh = np.load(mesh_dir + mesh_file + "_source.npz")
    target_mesh = np.load(mesh_dir + mesh_file + "_target.npz")

    tp = None
    sp_values = None

    connVal = source_mesh["connectivity"].astype(np.int64)
    connOff = source_mesh["offsets"].astype(np.int64)

    cellTypes = pacman.fe.vtk_to_pacman_cell_type(source_mesh["types"].astype(np.int64))

    spaceDimension = None

    if mesh_file == "aste":
        spaceDimension = 3
    else:
        spaceDimension = pacman.fe.vtk_cell_dim(int(source_mesh["types"][0]))

    sp = source_mesh["points"][:, :spaceDimension].astype(np.float64)
    tp = target_mesh["points"][:, :spaceDimension].astype(np.float64)

    if spaceDimension == 1:
        sp_values = franke_1d(sp[:,0])
    elif spaceDimension == 2:
        sp_values = franke_2d(sp[:,0], sp[:,1])
    else:
        sp_values = franke_3d(sp[:,0], sp[:,1], sp[:,2])

    tp_values, tp_status = pacman.fe.interpolate(spaceDimension, execspace, method, sp, sp_values, connVal, connOff, cellTypes, tp)

    tp_ref = sp_values

    if mesh_file == "aste":
        tp_ref = franke_3d(tp[:,0], tp[:,1], tp[:,2])
        error_norm = np.linalg.norm(tp_values - tp_ref) / np.linalg.norm(tp_ref)
        assert error_norm < 1e-3, f"Method {method} for mesh_file {mesh_file}: error norm {error_norm} exceeds threshold"
    else:
        if method == pacman.fe.methods.INTERP_ZEROFILL:
            assert np.all(tp_values[tp_status == 0] == 0), f"Interp/ZeroFill method for mesh_file {mesh_file}: expected 0 values where tp_status == 0"
            assert np.allclose(tp_values[tp_status != 0], tp_ref[tp_status != 0]), f"Method {method} for mesh_file {mesh_file}: tp_values and tp_ref do not match"
        elif method == pacman.fe.methods.INTERP_EXTRAP:
            #extrapolated values may differ
            assert np.allclose(tp_values[tp_status != 3], tp_ref[tp_status != 3]), f"Method {method} for mesh_file {mesh_file}: tp_values and tp_ref do not match"
        else:
            if not np.allclose(tp_values, tp_ref):
                print(f"tp_status: {tp_status}")
                print(f"tp_values: {tp_values}")
                print(f"tp_ref: {tp_ref}")
            assert np.allclose(tp_values, tp_ref), f"Method {method} for mesh_file {mesh_file}: tp_values and tp_ref do not match"

def main():
    """Parse CLI arguments and execute one FE interpolation test case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh")
    parser.add_argument("--method")
    parser.add_argument("--exec_space")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(script_dir, "./meshes/")

    methods = {
        "nearest_nearest": pacman.fe.methods.NEAREST_NEAREST,
        "interp_clamp": pacman.fe.methods.INTERP_CLAMP,
        "interp_nearest": pacman.fe.methods.INTERP_NEAREST,
        "interp_zerofill": pacman.fe.methods.INTERP_ZEROFILL,
        "interp_extrap": pacman.fe.methods.INTERP_EXTRAP
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

    test_interpolation(mesh_dir, args.mesh, methods[args.method], requested_execspace)

if __name__ == "__main__":
    main()