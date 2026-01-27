import os
import meshio
import numpy as np
import pacman
import argparse

def franke_2d(x, y):
    return (
        0.75
        * np.exp(
            -(
                ((9.0 * x - 2.0) * (9.0 * x - 2.0))
                + ((9.0 * y - 2.0) * (9.0 * y - 2.0))
            )
            / 4.0
        )
        + 0.75
        * np.exp(
            -(
                ((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                + (9.0 * y + 1.0) / 10.0
            )
        )
        + 0.5
        * np.exp(
            -(
                ((9.0 * x - 7.0) * (9.0 * x - 7.0))
                + ((9.0 * y - 3.0) * (9.0 * y - 3.0))
            ) / 4.0
        )
        - 0.2
        * np.exp(
            -(
                ((9.0 * x - 4.0) * (9.0 * x - 4.0))
                + ((9.0 * y - 7.0) * (9.0 * y - 7.0))
            )
        )
    )

def franke_3d(x, y, z):
    return (
        0.75
        * np.exp(
            -(
                ((9.0 * x - 2.0) * (9.0 * x - 2.0))
                + ((9.0 * y - 2.0) * (9.0 * y - 2.0))
                + ((9.0 * z - 2.0) * (9.0 * z - 2.0))
            )
            / 4.0
        )
        + 0.75
        * np.exp(
            -(
                ((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                + (9.0 * y + 1.0) / 10.0
                + (9.0 * z + 1.0) / 10.0
            )
        )
        + 0.5
        * np.exp(
            -(
                ((9.0 * x - 7.0) * (9.0 * x - 7.0))
                + ((9.0 * y - 3.0) * (9.0 * y - 3.0))
                + ((9.0 * z - 5.0) * (9.0 * z - 5.0))
            ) / 4.0
        )
        - 0.2
        * np.exp(
            -(
                ((9.0 * x - 4.0) * (9.0 * x - 4.0))
                + ((9.0 * y - 7.0) * (9.0 * y - 7.0))
                + ((9.0 * z - 5.0) * (9.0 * z - 5.0))
            )
        )
    )

def test_interpolation(mesh_dir, mesh_file, method, execspace):

    spaceDimension = None

    if mesh_file == "aste":
        spaceDimension = 3
    else:
        spaceDimension = pacman.fe.meshio_cell_dim(mesh_file)

    source_mesh = meshio.read(mesh_dir + mesh_file + "_source.vtk")
    target_mesh = meshio.read(mesh_dir + mesh_file + "_target.vtk")

    tp = None
    sp_values = None

    if spaceDimension == 2:
        sp = source_mesh.points[:,0:spaceDimension].astype(np.float64)
        tp = target_mesh.points[:,0:spaceDimension].astype(np.float64)
        sp_values = franke_2d(sp[:,0], sp[:,1])
    else:
        sp = source_mesh.points.astype(np.float64)
        tp = target_mesh.points.astype(np.float64)
        sp_values = franke_3d(sp[:,0], sp[:,1], sp[:,2])

    connVal = np.array([]).astype(np.int64).flatten()
    connOff = np.array([]).astype(np.int64).flatten()
    cellTypes = np.array([]).astype(np.int64).flatten()

    for eltype in source_mesh.cells_dict:
        if pacman.fe.meshio_cell_dim(eltype) < spaceDimension-1 :
            continue
        conn = source_mesh.cells_dict[eltype]
        if connVal.shape[0]==0:
            connVal = conn.flatten()
        else:
            np.hstack((connVal, conn.flatten()))
        loc_off = conn.shape[1]*np.ones(conn.shape[0])
        if connOff.shape[0]==0:
            connOff = loc_off.flatten()
        else:
            np.hstack((connOff, loc_off))
        vtk_cell_type_idx = pacman.fe.meshio_to_vtk_cell_type(eltype)
        loc_celltype = vtk_cell_type_idx*np.ones(conn.shape[0])
        if cellTypes.shape[0]==0:
            cellTypes = loc_celltype.flatten()
        else:
            np.hstack((connOff, loc_off))

    connOff = np.cumsum(connOff).astype(np.int64)
    connOff = np.insert(connOff, 0, 0)

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
        "serial": pacman.execspaces.SERIAL,
        "openmp": pacman.execspaces.OPENMP,
        "cuda": pacman.execspaces.CUDA
    }

    test_interpolation(mesh_dir, args.mesh, methods[args.method], execspaces[args.exec_space])


if __name__ == "__main__":
    main()