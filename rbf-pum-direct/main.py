import numpy as np
import os
import RbfPumInterpolator
import meshio
from time import time_ns

def franke_2d(x, y):
    return (
    0.75 * np.exp(-((9*x - 2)**2 + (9*y - 2)**2)/4.0)
    + 0.75 * np.exp(-((9*x + 1)**2)/49.0 - (9*y + 1)/10.0)
    + 0.5  * np.exp(-((9*x - 7)**2 + (9*y - 3)**2)/4.0)
    - 0.2  * np.exp(-((9*x - 4)**2 + (9*y - 7)**2))
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
                + ((9.0 * z - 5.0) * (9.0 * z - 5.0)) / 4.0
            )
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

def generate_2d_grid(xmin, xmax, ymin, ymax, nx, ny):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    pts = np.column_stack((X.ravel(), Y.ravel()))
    return pts

def test_2d(source_bbox, target_bbox, source_n, target_n):
    sxmin, sxmax, symin, symax = source_bbox
    txmin, txmax, tymin, tymax = target_bbox
    nx_s, ny_s = source_n
    nx_t, ny_t = target_n

    source_pts = generate_2d_grid(sxmin, sxmax, symin, symax, nx_s, ny_s)
    target_pts = generate_2d_grid(txmin, txmax, tymin, tymax, nx_t, ny_t)

    src_x, src_y = source_pts[:, 0], source_pts[:, 1]
    tgt_x, tgt_y = target_pts[:, 0], target_pts[:, 1]

    source_values = franke_2d(src_x, src_y)
    reference = franke_2d(tgt_x, tgt_y)

    t1 = time_ns()
    out = RbfPumInterpolator.interpolate(source_pts, source_values, target_pts)
    t2 = time_ns()

    errors = np.abs(reference - out)

    print(f"Total Python time (2d case): {(t2 - t1) / 1000000.0} ms")
    print(f"mean abs error: {np.mean(errors)}")
    print(f"max abs error:  {np.max(errors)}")

def test_3d(source_mesh_file, target_mesh_file):
    source_pts, _ = read_mesh(source_mesh_file)
    target_pts, _ = read_mesh(target_mesh_file)
    source_values = franke_3d(source_pts[:,0], source_pts[:,1], source_pts[:,2])
    reference = franke_3d(target_pts[:,0], target_pts[:,1], target_pts[:,2])
    t1 = time_ns()
    out = RbfPumInterpolator.interpolate(source_pts, source_values, target_pts)
    t2 = time_ns()
    errors = np.abs(reference - out)

    print(f"Total Python time (3d case): {(t2 - t1) / 1000000.0} ms")
    print(f'mean abs error: {np.mean(errors)}\nmax abs error:  {np.max(errors)}')

def main():
    source_bb = (0.0, 1.0, 0.0, 1.0)
    target_bb = (0.0, 1.0, 0.0, 1.0)
    ds = (100, 100)
    dt = (1000, 1000)
    test_2d(source_bb, target_bb, ds, dt)

    source_mesh_file = "./meshes/0.01.vtu"
    target_mesh_file = "./meshes/0.001.vtu"
    test_3d(source_mesh_file, target_mesh_file)

if __name__ == "__main__":
    main()
