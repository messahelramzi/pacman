import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import yaml
import os


def main():
    yaml_files = list(Path("data").glob("*.yaml"))
    yaml_files.sort()
    methods_names = list(
        map(
            lambda elt: str(elt.parts[-1]).replace("result_", "").replace(".yaml", ""),
            yaml_files,
        )
    )
    output_dir = "output_graphs"
    if not os.path.exists(f"{output_dir}/"):
        os.makedirs(f"{output_dir}/")

    average = []
    min = []
    max = []
    median = []
    percentile_90 = []
    percentile_95 = []
    percentile_99 = []
    time_ms = []

    stats = {
        "average": average,
        "min": min,
        "max": max,
        "median": median,
        "90%": percentile_90,
        "95%": percentile_95,
        "99%": percentile_99,
        "time-ms": time_ms,
    }

    colors = {
        "P0P0": "tab:orange",
        "P0P1": "tab:orange",
        "P1P0": "tab:orange",
        "P1P1": "tab:orange",
        "interp-clamp": "tab:purple",
        "interp-extrap": "tab:purple",
        "interp-nearest": "tab:purple",
        "interp-zerofill": "tab:purple",
        "nearest-nearest": "tab:purple",
        "rbf-pum-direct": "tab:blue",
        "rbf-global-direct": "tab:blue",
        "nearest-neighbor": "tab:blue",
        "nearest-neighbor-projection": "tab:blue",
    }

    for file in yaml_files:
        f = open(file, "r")
        main_node = yaml.safe_load(f)
        average.append(main_node["values"]["avg"])
        min.append(main_node["values"]["min"])
        max.append(main_node["values"]["max"])
        median.append(main_node["values"]["med"])
        percentile_90.append(main_node["values"]["p90"])
        percentile_95.append(main_node["values"]["p95"])
        percentile_99.append(main_node["values"]["p99"])
        time_ms.append(main_node["values"]["time-ms"])
        f.flush()
        f.close()

    bar_colors = []
    for method in methods_names:
        bar_colors.append(colors[method])

    # bar graphs
    for name, value in stats.items():
        y = value
        x = np.arange(len(methods_names)) * 1.5
        plt.figure()
        bars = plt.bar(x, y, width=1, color=bar_colors)
        plt.xticks(x, methods_names, rotation=45, ha="right")
        plt.yscale("log")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if name == "time-ms":
            plt.bar_label(bars, labels=[f"{int(v)}" for v in y], padding=0.1)
            plt.title("Interpolation Matrix Computation Time (ms)")
        else:
            plt.title(f"{name} errors")
        plt.savefig(f"{output_dir}/{name}.svg", format="svg")


if __name__ == "__main__":
    main()
