#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import constants

RESULTS_ROOT = constants.CONTINUOUS_RESULTS_DIR
PLOTS_DIR = os.path.join(RESULTS_ROOT, "plots")
OUTPUT_STEM = os.path.join(PLOTS_DIR, "social_fusion_rule_comparison")
SWARM_SIZE = 12
TERRAINS = constants.TERRAIN_NAMES
METHODS = (
    ("comp_boost", "Compatibility boost", "#1f4e79", -0.05),
    ("yager", "Yager's rule", "#c04b32", 0.05),
)


def load_trial_rows(experiment_name):
    path = os.path.join(RESULTS_ROOT, "swarm_size", experiment_name, "trial_summary.csv")
    with open(path, newline = "", encoding = "utf-8") as handle:
        rows = list(csv.DictReader(handle))

    deduplicated = {}
    for row in rows:
        deduplicated[(row["terrain"], row["trial_index"])] = row
    return list(deduplicated.values())


def parse_bool(value):
    return str(value).strip().lower() == "true"


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def collect_metric_values(rows, metric_name):
    return [
        parse_float(row[metric_name])
        for row in rows
        if not parse_bool(row["timeout_reached"]) and parse_float(row[metric_name]) is not None
    ]


def build_summary(terrains):
    summary = defaultdict(dict)
    timeout_percentages = {}

    for method_key, _, _, _ in METHODS:
        experiment_name = f"swarm_size_{SWARM_SIZE}_{method_key}"
        rows = load_trial_rows(experiment_name)

        terrain_groups = defaultdict(list)
        for row in rows:
            terrain_groups[row["terrain"]].append(row)

        for terrain in terrains:
            terrain_rows = terrain_groups.get(terrain, [])
            summary["position_error_meters"][(terrain, method_key)] = collect_metric_values(
                terrain_rows,
                "position_error_meters",
            )
            summary["consensus_time"][(terrain, method_key)] = collect_metric_values(
                terrain_rows,
                "consensus_time",
            )

            timeout_count = sum(1 for row in terrain_rows if parse_bool(row["timeout_reached"]))
            timeout_percentages[(terrain, method_key)] = 100.0 * timeout_count / len(terrain_rows) if terrain_rows else 0.0

    return summary, timeout_percentages


def compute_ylim(value_groups, label_offset = 3.0):
    maxima = [max(values) for values in value_groups if values]
    if not maxima:
        return (0.0, 1.0)

    upper = max(maxima)
    # Leave headroom for timeout annotations and whiskers.
    return (0.0, upper * 1.12 + label_offset)


def style_boxplot(boxplot, color):
    for box in boxplot["boxes"]:
        box.set(facecolor = color, alpha = 0.35, edgecolor = color, linewidth = 1.4)
    for median in boxplot["medians"]:
        median.set(color = color, linewidth = 2.2)
    for whisker in boxplot["whiskers"]:
        whisker.set(color = color, linewidth = 1.2)
    for cap in boxplot["caps"]:
        cap.set(color = color, linewidth = 1.2)


def upper_whisker(values):
    values_array = np.asarray(values, dtype = np.float64)
    q1 = float(np.percentile(values_array, 25))
    q3 = float(np.percentile(values_array, 75))
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    non_outlier_values = values_array[values_array <= upper_threshold]
    return float(np.max(non_outlier_values)) if len(non_outlier_values) else float(np.max(values_array))


def plot_social_fusion_rule_comparison(terrains = TERRAINS, output_stem = OUTPUT_STEM):
    os.makedirs(PLOTS_DIR, exist_ok = True)
    summary, timeout_percentages = build_summary(terrains)

    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    figure, axes = plt.subplots(2, 1, figsize = (5, 7.2), sharex = True)
    figure.subplots_adjust(left = 0.12, right = 0.88, top = 0.98, bottom = 0.12, hspace = 0.14)

    x_positions = np.arange(1, len(terrains) + 1, dtype = np.float64) * 0.4
    error_axis, time_axis = axes
    yager_error_axis = error_axis.twinx()

    error_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)
    time_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)

    all_time_values = []

    for method_key, method_label, color, offset in METHODS:
        shifted_x = x_positions + offset
        error_values = [summary["position_error_meters"][(terrain, method_key)] for terrain in terrains]
        time_values = [summary["consensus_time"][(terrain, method_key)] for terrain in terrains]
        all_time_values.extend(time_values)

        current_error_axis = error_axis if method_key == "comp_boost" else yager_error_axis

        error_boxplot = current_error_axis.boxplot(
            error_values,
            positions = shifted_x,
            widths = 0.05,
            patch_artist = True,
            showfliers = False,
            manage_ticks = False,
        )
        time_boxplot = time_axis.boxplot(
            time_values,
            positions = shifted_x,
            widths = 0.05,
            patch_artist = True,
            showfliers = False,
            manage_ticks = False,
        )
        style_boxplot(error_boxplot, color)
        style_boxplot(time_boxplot, color)
        error_axis.plot([], [], color = color, linewidth = 8, alpha = 0.35, label = method_label)

        for x_coord, terrain, values in zip(shifted_x, terrains, time_values):
            if not values:
                continue
            time_axis.annotate(
                f"{timeout_percentages[(terrain, method_key)]:.0f}%",
                (x_coord, upper_whisker(values)),
                xytext = (0.5, 3),
                textcoords = "offset points",
                ha = "center",
                va = "bottom",
                fontsize = 7.5,
                color = color,
                alpha = 0.9,
            )

    error_axis.set_ylabel("Position error (m)", color = METHODS[0][2])
    yager_error_axis.set_ylabel("Position error (m)", color = METHODS[1][2])
    time_axis.set_ylabel("Consensus time (s)")
    time_axis.set_xticks(x_positions, [terrain.capitalize() for terrain in terrains])
    figure.supxlabel("Terrain", y = 0.045, fontsize = 11)
    figure.align_ylabels(axes)

    error_axis.set_ylim(0.0, 9000.0)
    yager_error_axis.set_ylim(0.0, 16000.0)
    time_axis.set_ylim(*compute_ylim(all_time_values, label_offset = 5.0))
    error_axis.tick_params(axis = "y", colors = METHODS[0][2])
    yager_error_axis.tick_params(axis = "y", colors = METHODS[1][2])

    time_axis.legend(
        *error_axis.get_legend_handles_labels(),
        loc = "upper right",
        frameon = True,
        facecolor = "white",
        edgecolor = "#cccccc",
        ncol = 1,
        fontsize = 8.5,
        borderpad = 0.35,
        labelspacing = 0.35,
        handlelength = 1.4,
    )

    png_path = output_stem + ".png"
    pdf_path = output_stem + ".pdf"
    figure.savefig(png_path, bbox_inches = "tight")
    figure.savefig(pdf_path, bbox_inches = "tight")
    plt.close(figure)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    plot_social_fusion_rule_comparison()
