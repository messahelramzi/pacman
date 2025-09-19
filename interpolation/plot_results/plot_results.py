from pathlib import Path

from requests import get
import yaml
import re
import numpy as np

import matplotlib.pyplot as plt


def get_medcoupling_reports(pair=None, method=None):
    if pair is not None:
        if method is not None:
            files = list(Path("data/").glob(f"result_{method}_pair{pair}.yaml"))
        else:
            files = list(Path("data/").glob(f"result_P*_pair{pair}.yaml"))
    elif method is not None:
        files = list(Path("data/").glob(f"result_{method}_pair*.yaml"))
    else:
        files = list(Path("data/").glob("result_P*.yaml"))
    return sorted(files)


def get_precice_reports(pair=None, method=None):
    if pair is not None:
        if method is not None:
            files = list(Path("data/").glob(f"result_{method}_pair{pair}.yaml"))
        else:
            files = list(Path("data/").glob(f"result_rbf-*_pair{pair}.yaml"))
            files += list(
                Path("data/").glob(f"result_nearest-neighbor*_pair{pair}.yaml")
            )
    elif method is not None:
        files = list(Path("data/").glob(f"result_{method}_pair*.yaml"))
    else:
        files = list(Path("data/").glob("result_rbf-*.yaml"))
        files += list(Path("data/").glob("result_nearest-neighbor*.yaml"))
    return sorted(files)


def get_muscat_reports(pair=None, method=None):
    if pair is not None:
        if method is not None:
            files = list(Path("data/").glob(f"result_{method}_pair{pair}.yaml"))
        else:
            files = list(Path("data/").glob(f"result_interp*_pair{pair}.yaml"))
            files += list(
                Path("data/").glob(f"result_nearest-nearest*_pair{pair}.yaml")
            )
    elif method is not None:
        files = list(Path("data/").glob(f"result_{method}_pair*.yaml"))
    else:
        files = list(Path("data/").glob("result_interp*.yaml"))
        files += list(Path("data/").glob("result_nearest-nearest*.yaml"))
    return sorted(files)


def transform_to_methods_names(reports):
    return list(
        map(
            lambda elt: re.sub(r".*/result_(.*)_pair[0123].yaml", r"\1", str(elt)),
            reports,
        )
    )


def get_avg_error_from_files(files):
    ret = []
    for file in files:
        f = open(file, "r", encoding="UTF-8")
        node = yaml.safe_load(f)
        ret.append(node["values"]["avg"])
        f.close()
    return ret


def get_time_from_files(files):
    ret = []
    for file in files:
        f = open(file, "r", encoding="UTF-8")
        node = yaml.safe_load(f)
        ret.append(node["values"]["time-ms"])
        f.close()
    return ret


def init_figure(cols, rows):
    plot_cols = cols
    plot_rows = rows

    subplot_min_width = 600
    subplot_min_height = 400
    dpi = 100

    figsize_width = (plot_cols * subplot_min_width) / dpi
    figsize_height = (plot_rows * subplot_min_height) / dpi

    fig, axes = plt.subplots(
        plot_rows, plot_cols, figsize=(figsize_width, figsize_height)
    )

    return fig, axes


def plot_avg_accuracy_per_pair():
    output_dir = "output_graphs"

    fig, axes = init_figure(4, 4)

    pairs_names = ["0.03 -> 0.003", "0.02 -> 0.002", "0.01 -> 0.001", "0.004 -> 0.0005"]
    libs = ["MEDCoupling", "preCICE", "Muscat"]
    for i, lib in enumerate(libs):
        for pair, pair_name in enumerate(pairs_names):
            files = []
            if i == 0:
                files += get_medcoupling_reports(pair=pair)
            if i == 1:
                files += get_precice_reports(pair=pair)
            if i == 2:
                files += get_muscat_reports(pair=pair)
            avg_error = get_avg_error_from_files(files)
            y = np.array(avg_error)
            x = np.arange(y.shape[0]) * 1.5
            current_subplot = axes.flat[i * 4 + pair]
            bars = current_subplot.bar(
                x,
                y,
                width=1,
                color=f"tab:{'orange' if i == 0 else 'blue' if i == 1 else 'purple'}",
            )
            current_subplot.set_xticks(
                x,
                transform_to_methods_names(files),
                rotation=45,
                ha="right",
            )
            current_subplot.set_yscale("log")
            current_subplot.set_title(
                f"{i * 4 + pair + 1}. {lib} {pair_name} avg error"
            )

    ii = 13
    # times
    for pair, pair_name in enumerate(pairs_names):
        medcoupling_files = get_medcoupling_reports(pair=pair)
        precice_files = get_precice_reports(pair=pair)
        muscat_files = get_muscat_reports(pair=pair)

        medcoupling_times = get_time_from_files(medcoupling_files)
        precice_times = get_time_from_files(precice_files)
        muscat_times = get_time_from_files(muscat_files)

        current_subplot = axes.flat[12 + pair]
        med_y = np.array(medcoupling_times)
        med_x = np.arange(med_y.shape[0]) * 1.5

        pre_y = np.array(precice_times)
        pre_x = (np.arange(pre_y.shape[0]) + med_y.shape[0]) * 1.5

        mus_y = np.array(muscat_times)
        mus_x = (np.arange(mus_y.shape[0]) + med_y.shape[0] + pre_y.shape[0]) * 1.5

        current_subplot.bar(med_x, med_y, width=1, color="tab:orange")
        current_subplot.bar(pre_x, pre_y, width=1, color="tab:blue")
        current_subplot.bar(mus_x, mus_y, width=1, color="tab:purple")
        xs = med_x.tolist() + pre_x.tolist() + mus_x.tolist()
        names = transform_to_methods_names(
            medcoupling_files + precice_files + muscat_files
        )
        current_subplot.set_xticks(xs, names, rotation=45, ha="right")
        current_subplot.set_yscale("log")
        current_subplot.set_title(f"{ii}. {pair_name} interpolation time (ms)")
        ii += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/global_stats.svg", format="svg")


def plot_diff_avg_time_precice_muscat():
    output_dir = "output_graphs"

    fig, axes = init_figure(2, 1)
    pairs_names = ["0.03 -> 0.003", "0.02 -> 0.002", "0.01 -> 0.001", "0.004 -> 0.0005"]

    precice_files = get_precice_reports(method="rbf-pum-direct")
    muscat_files = get_muscat_reports(method="interp-clamp")

    precice_avg_err = get_avg_error_from_files(precice_files)
    muscat_avg_err = get_avg_error_from_files(muscat_files)

    precice_avg_time = get_time_from_files(precice_files)
    muscat_avg_time = get_time_from_files(muscat_files)

    err_subplot = axes.flat[0]
    time_subplot = axes.flat[1]

    precice_avg_err_y = np.array(precice_avg_err)
    muscat_avg_err_y = np.array(muscat_avg_err)
    avg_err_x = (
        np.arange(max(precice_avg_err_y.shape[0], muscat_avg_err_y.shape[0])) * 1.5
    )

    err_subplot.plot(
        avg_err_x,
        precice_avg_err_y,
        color="tab:blue",
        label="preCICE",
        marker="o",
        markersize=4.0,
    )
    err_subplot.plot(
        avg_err_x,
        muscat_avg_err_y,
        color="tab:purple",
        label="Muscat",
        marker="o",
        markersize=4.0,
    )

    err_subplot.legend()

    err_subplot.set_yscale("log")

    precice_time_y = np.array(precice_avg_time)
    muscat_time_y = np.array(muscat_avg_time)
    time_x = np.arange(max(precice_time_y.shape[0], muscat_time_y.shape[0])) * 1.5

    time_subplot.plot(
        time_x,
        precice_time_y,
        color="tab:blue",
        label="preCICE",
        marker="o",
        markersize=4.0,
    )
    time_subplot.plot(
        time_x,
        muscat_time_y,
        color="tab:purple",
        label="Muscat",
        marker="o",
        markersize=4.0,
    )

    time_subplot.legend()

    # time_subplot.set_yscale("log")

    err_subplot.set_xticks(avg_err_x, pairs_names, rotation=45, ha="right")
    time_subplot.set_xticks(time_x, pairs_names, rotation=45, ha="right")

    err_subplot.set_title("1. Average Error Over Mesh Refinement Level")
    time_subplot.set_title("2. Interpolation Matrix Average Computation Time (ms)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/line_charts.svg", format="svg")


def main():
    plot_avg_accuracy_per_pair()
    plot_diff_avg_time_precice_muscat()
    return


if __name__ == "__main__":
    main()
