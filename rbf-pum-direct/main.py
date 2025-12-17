import numpy as np
import os
import interpolator
import meshio
from time import time_ns

def franke_1d(x):
    return (
        0.75 * np.exp(-((9*x - 2)**2)/4.0)
        + 0.75 * np.exp(-((9*x + 1)**2)/49.0)
        + 0.5  * np.exp(-((9*x - 7)**2)/4.0)
        - 0.2  * np.exp(-((9*x - 4)**2))
    )

def franke_2d(x, y):
    return (
        0.75 * np.exp(-((9*x - 2)**2 + (9*y - 2)**2)/4.0)
        + 0.75 * np.exp(-((9*x + 1)**2)/49.0 - (9*y + 1)/10.0)
        + 0.5  * np.exp(-((9*x - 7)**2 + (9*y - 3)**2)/4.0)
        - 0.2  * np.exp(-((9*x - 4)**2 + (9*y - 7)**2))
    )

def franke_3d(x, y, z):
    return (
        0.75 * np.exp(-(((9.0 * x - 2.0) * (9.0 * x - 2.0)) + ((9.0 * y - 2.0) * (9.0 * y - 2.0)) + ((9.0 * z - 2.0) * (9.0 * z - 2.0))) / 4.0)
        + 0.75 * np.exp(-(((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0 + (9.0 * y + 1.0) / 10.0 + (9.0 * z + 1.0) / 10.0))
        + 0.5 * np.exp(-(((9.0 * x - 7.0) * (9.0 * x - 7.0)) + ((9.0 * y - 3.0) * (9.0 * y - 3.0)) + ((9.0 * z - 5.0) * (9.0 * z - 5.0))) / 4.0)
        - 0.2 * np.exp(-(((9.0 * x - 4.0) * (9.0 * x - 4.0)) + ((9.0 * y - 7.0) * (9.0 * y - 7.0)) + ((9.0 * z - 5.0) * (9.0 * z - 5.0))))
    )

def franke_function(values):
    d = values.shape[1]
    if (d == 1):
        return franke_1d(values[:, 0])
    if (d == 2):
        return franke_2d(values[:, 0], values[:, 1])
    if (d == 3):
        return franke_3d(values[:, 0], values[:, 1], values[:, 2])
    raise RuntimeError()

def read_mesh(filename: str):
    if (not os.path.exists(filename)):
        raise RuntimeError(f"{filename} does not match any file")
    mesh = meshio.read(filename)
    pts = np.asarray(mesh.points, dtype=np.float64)
    if mesh.point_data:
        first_key = next(iter(mesh.point_data))
        vals = np.asarray(mesh.point_data[first_key], dtype=np.float64)
    else:
        vals = None
    return pts, vals

def test_2d(source_mesh_file, target_mesh_file, execspace, rbf_function):
    source_pts, _ = read_mesh(source_mesh_file)
    source_pts = source_pts[:, 0:2]
    target_pts, _ = read_mesh(target_mesh_file)
    target_pts = target_pts[:, 0:2]
    source_values = franke_function(source_pts)
    reference = franke_function(target_pts)
    t1 = time_ns()
    out = interpolator.rbf.interpolate(source_pts, source_values, target_pts, execspace, rbf_function)
    t2 = time_ns()
    errors = np.abs(reference - out)

    print(f"Python time (2d case): {(t2 - t1) / 1000000.0} ms")
    print(f'mean abs error: {np.mean(errors)}')
    print(f'max abs error:  {np.max(errors)}')

def test_3d(source_mesh_file, target_mesh_file, execspace, rbf_function):
    source_pts, _ = read_mesh(source_mesh_file)
    target_pts, _ = read_mesh(target_mesh_file)
    source_values = franke_function(source_pts)
    reference = franke_function(target_pts)
    t1 = time_ns()
    out = interpolator.rbf.interpolate(source_pts, source_values, target_pts, execspace, rbf_function)
    t2 = time_ns()
    errors = np.abs(reference - out)

    print(f"Python time (3d case): {(t2 - t1) / 1000000.0} ms")
    print(f'mean abs error: {np.mean(errors)}')
    print(f'max abs error:  {np.max(errors)}')

def main():
    folder_2d = "./meshes/2d"
    folder_3d = "./meshes/3d"
    source = "0.004.vtu"
    target = "0.0005.vtu"
    backend = interpolator.execspaces.CUDA
    function = interpolator.rbf.functions.WENDLANDC6
    test_3d(f'{folder_3d}/{source}', f'{folder_3d}/{target}', backend, function)

    backend = interpolator.execspaces.CUDA
    function = interpolator.rbf.functions.WENDLANDC6
    test_2d(f'{folder_2d}/{source}', f'{folder_2d}/{target}', backend, function)


if __name__ == "__main__":
    main()
