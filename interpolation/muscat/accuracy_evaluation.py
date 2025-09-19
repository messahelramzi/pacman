import os
import time
from pathlib import Path

import numpy as np
from Muscat.FE.Fields.FEField import FEField
from Muscat.IO.UniversalReader import InitAllReaders, ReadMesh
from Muscat.IO.XdmfWriter import WriteMeshToXdmf
from Muscat.MeshTools.MeshFieldOperations import GetFieldTransferOpCpp
from Muscat.MeshTools.MeshMappingTools import PrepareFEComputation


def franke_function(x, y, z):
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


def eval_interp(source_mesh_path, target_mesh_path, interp_mode, output_path):
    input_mesh = ReadMesh(source_mesh_path)
    franke_data = franke_function(
        input_mesh.nodes[:, 0], input_mesh.nodes[:, 1], input_mesh.nodes[:, 2]
    )
    input_field = FEField("FrankeData", input_mesh, None, None, franke_data)

    target_mesh = ReadMesh(target_mesh_path)
    t1 = time.perf_counter_ns()
    P, _, _ = GetFieldTransferOpCpp(input_field, target_mesh.nodes, method=interp_mode)
    t2 = time.perf_counter_ns()
    interpolated_data = P.dot(franke_data)
    # target_space, target_numberings, _, _ = PrepareFEComputation(
    #     target_mesh, numberOfComponents=1
    # )

    target_reference_data = franke_function(
        target_mesh.nodes[:, 0], target_mesh.nodes[:, 1], target_mesh.nodes[:, 2]
    )
    error_data = np.abs(interpolated_data - target_reference_data)

    # WriteMeshToXdmf(
    #     f"{output_path}_{interp_mode.replace('/', '-').lower()}.xdmf",
    #     target_mesh,
    #     PointFields=[target_reference_data, interpolated_data, error_data],
    #     PointFieldsNames=["FrankeData", "InterpolatedData", "ErrorData"],
    # )

    return error_data, (t2 - t1) / 1000000.0


def main():
    meshes_pairs = [
        ("0.03", "0.003"),
        ("0.02", "0.002"),
        ("0.01", "0.001"),
        ("0.004", "0.0005"),
    ]
    interpolation_modes = [
        "Interp/Nearest",
        "Nearest/Nearest",
        "Interp/Clamp",
        "Interp/Extrap",
        "Interp/ZeroFill",
    ]

    output_file = "result"
    output_folder = Path("../plot_results/data")

    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    except Exception as e:
        raise e

    output_path = f"{output_folder}/{output_file}"

    for nb_p, p in enumerate(meshes_pairs):
        error_fields = []
        times = []
        source_mesh_path = "meshes" + "/" + p[0] + ".vtk"
        target_mesh_path = "meshes" + "/" + p[1] + ".vtk"
        for it in interpolation_modes:
            f, t = eval_interp(source_mesh_path, target_mesh_path, it, output_path)
            error_fields.append(f)
            times.append(t)
        # We compute time spent to compute the interpolation matrix over 10 iterations
        for _ in range(9):
            for it in interpolation_modes:
                _, t = eval_interp(source_mesh_path, target_mesh_path, it, output_path)
                times.append(t)

        print(len(times))
        times2 = np.array(times)
        for i, field in enumerate(error_fields):
            sorted_abs_data = np.sort(np.abs(field))

            report = "values:\n"
            report += f"    avg: {np.mean(sorted_abs_data)}\n"
            report += f"    min: {np.min(sorted_abs_data)}\n"
            report += f"    max: {np.max(sorted_abs_data)}\n"
            report += f"    med: {sorted_abs_data[sorted_abs_data.shape[0] // 2]}\n"
            report += (
                f"    p90: {sorted_abs_data[int(sorted_abs_data.shape[0] * 0.90)]}\n"
            )
            report += (
                f"    p95: {sorted_abs_data[int(sorted_abs_data.shape[0] * 0.95)]}\n"
            )
            report += (
                f"    p99: {sorted_abs_data[int(sorted_abs_data.shape[0] * 0.99)]}\n"
            )
            report += f"    time-ms: {np.mean(times2[i::5])}\n"
            report += "    invalid-values: false\n"

            output_file = open(
                f"{output_path}_{interpolation_modes[i].replace('/', '-').lower()}_pair{nb_p}.yaml",
                "w",
            )
            output_file.write(report)
            output_file.flush()
            output_file.close()


if __name__ == "__main__":
    InitAllReaders()
    main()
