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

    spaceDimension = 3
    source_mesh = meshio.read(mesh_dir + mesh_file[0])
    target_mesh = meshio.read(mesh_dir + mesh_file[1])

    tp = None
    sp_values = None

    sp = source_mesh.points.astype(np.float64)
    tp = target_mesh.points.astype(np.float64)
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
        "serial": pacman.execspaces.SERIAL,
        "openmp": pacman.execspaces.OPENMP,
        "cuda": pacman.execspaces.CUDA
    }

    meshes= {
        # Coarse to Fine
        "coarse_to_fine_0": ("0.03.vtu", "0.003.vtu"),
        "coarse_to_fine_1": ("0.02.vtu", "0.002.vtu"),
        "coarse_to_fine_2": ("0.01.vtu", "0.001.vtu"),
        "coarse_to_fine_3": ("0.004.vtu", "0.0005.vtu"),

        # Fine to Coarse
        "fine_to_coarse_0": ("0.003.vtu", "0.03.vtu"),
        "fine_to_coarse_1": ("0.002.vtu", "0.02.vtu"),
        "fine_to_coarse_2": ("0.001.vtu", "0.01.vtu"),
        "fine_to_coarse_3": ("0.0005.vtu", "0.004.vtu"),

        # Same meshes
        "same_0": ("0.03.vtu", "0.03.vtu"),
        "same_1": ("0.001.vtu", "0.001.vtu"),
        "same_2": ("0.0005.vtu", "0.0005.vtu"),
    }

    test_interpolation(mesh_dir, meshes[args.mesh], methods[args.method], execspaces[args.exec_space])


if __name__ == "__main__":
    main()