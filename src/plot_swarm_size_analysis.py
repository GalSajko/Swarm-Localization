#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = "results/continuous_est"
PLOTS_DIR = os.path.join(RESULTS_ROOT, "plots")
OUTPUT_STEM = os.path.join(PLOTS_DIR, "swarm_size_analysis")
FIGURE_TITLE = "Effect of Swarm Size on Localization Error and Consensus Time for Compatibility Boost and Yager's Rule"
TERRAINS = ("city", "desert", "mountains", "rural")
SWARM_SIZES = (2, 4, 9, 12, 16, 20, 25)
# Per-method visual settings:
# color and horizontal offset control how far apart the two boxplots are at each swarm-size tick.
METHODS = (
    ("comp_boost", "Compatibility boost", "#4f7ea4", -0.18),
    ("yager", "Yager's rule", "#c6726b", 0.18),
)

# Optional per-label tweaks for timeout annotations.
# Key format: (terrain, swarm_size, method_key)
# Value: (x_offset_points, y_offset_points)
TIMEOUT_LABEL_OFFSETS = {
    ("rural", 2, "yager"): (0.5, 8),
}


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


def build_summary():
    summary = defaultdict(dict)
    timeout_percentages = defaultdict(dict)

    for swarm_size in SWARM_SIZES:
        for method_key, _, _, _ in METHODS:
            experiment_name = f"swarm_size_{swarm_size}_{method_key}"
            rows = load_trial_rows(experiment_name)

            terrain_groups = defaultdict(list)
            for row in rows:
                terrain_groups[row["terrain"]].append(row)

            for terrain in TERRAINS:
                terrain_rows = terrain_groups.get(terrain, [])
                summary[(terrain, "position_error_meters")][(swarm_size, method_key)] = collect_metric_values(
                    terrain_rows,
                    "position_error_meters",
                )
                summary[(terrain, "consensus_time")][(swarm_size, method_key)] = collect_metric_values(
                    terrain_rows,
                    "consensus_time",
                )

                timeout_count = sum(1 for row in terrain_rows if parse_bool(row["timeout_reached"]))
                timeout_percentages[terrain][(swarm_size, method_key)] = (
                    100.0 * timeout_count / len(terrain_rows) if terrain_rows else 0.0
                )

    return summary, timeout_percentages


def plot_swarm_size_analysis():
    os.makedirs(PLOTS_DIR, exist_ok = True)
    summary, timeout_percentages = build_summary()

    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    figure, axes = plt.subplots(
        2,
        len(TERRAINS),
        # Main figure size knob. Increase height if timeout labels feel cramped.
        figsize = (16, 8.5),
        sharex = True,
    )
    # Layout knobs:
    # - wspace controls horizontal gaps between terrain columns
    # - hspace controls vertical gap between error/time rows
    # - left/right/top/bottom trim outer whitespace
    figure.subplots_adjust(left = 0.07, right = 0.995, top = 0.90, bottom = 0.12, wspace = 0.25, hspace = 0.16)

    x_positions = np.arange(1, len(SWARM_SIZES) + 1, dtype = np.float64)

    def style_boxplot(boxplot, color):
        # Boxplot styling knobs. Adjust alpha/linewidth/markersize here for readability.
        for box in boxplot["boxes"]:
            box.set(facecolor = color, alpha = 0.35, edgecolor = color, linewidth = 1.4)
        for median in boxplot["medians"]:
            median.set(color = color, linewidth = 2.2)
        for whisker in boxplot["whiskers"]:
            whisker.set(color = color, linewidth = 1.2)
        for cap in boxplot["caps"]:
            cap.set(color = color, linewidth = 1.2)
        for flier in boxplot["fliers"]:
            flier.set(
                marker = "o",
                markersize = 2.4,
                markerfacecolor = color,
                markeredgecolor = color,
                alpha = 0.35,
            )

    # Shared y-axis limits across terrains.
    # Adjust these manually when you want a fixed poster scale.
    error_ylim = (0.0, 18000.0)
    time_ylim = (0.0, 400.0)

    for terrain_index, terrain in enumerate(TERRAINS):
        error_axis = axes[0, terrain_index]
        time_axis = axes[1, terrain_index]

        error_axis.set_title(terrain.capitalize())
        error_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)
        time_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)
        error_axis.set_ylim(*error_ylim)
        time_axis.set_ylim(*time_ylim)

        for method_key, method_label, color, offset in METHODS:
            # Offset keeps comp_boost and yager boxplots side by side instead of overlapping.
            shifted_x = x_positions + offset
            error_values = [
                summary[(terrain, "position_error_meters")][(swarm_size, method_key)]
                for swarm_size in SWARM_SIZES
            ]
            time_values = [
                summary[(terrain, "consensus_time")][(swarm_size, method_key)]
                for swarm_size in SWARM_SIZES
            ]

            error_boxplot = error_axis.boxplot(
                error_values,
                positions = shifted_x,
                widths = 0.28,
                patch_artist = True,
                showfliers = False,
                manage_ticks = False,
            )
            time_boxplot = time_axis.boxplot(
                time_values,
                positions = shifted_x,
                widths = 0.28,
                patch_artist = True,
                showfliers = False,
                manage_ticks = False,
            )
            style_boxplot(error_boxplot, color)
            style_boxplot(time_boxplot, color)

            if terrain_index == 0:
                error_axis.plot([], [], color = color, linewidth = 8, alpha = 0.35, label = method_label)

            for x_coord, values, swarm_size in zip(shifted_x, time_values, SWARM_SIZES):
                if not values:
                    continue
                values_array = np.asarray(values, dtype = np.float64)
                q1 = float(np.percentile(values_array, 25))
                q3 = float(np.percentile(values_array, 75))
                iqr = q3 - q1
                upper_threshold = q3 + 1.5 * iqr
                non_outlier_values = values_array[values_array <= upper_threshold]
                top_whisker = float(np.max(non_outlier_values)) if len(non_outlier_values) else float(np.max(values_array))
                timeout_percentage = timeout_percentages[terrain][(swarm_size, method_key)]
                time_axis.annotate(
                    f"{timeout_percentage:.0f}%",
                    (x_coord, top_whisker),
                    # Timeout label offset:
                    # increase the second value if labels touch whiskers,
                    # decrease it if labels float too high above the plots.
                    xytext = TIMEOUT_LABEL_OFFSETS.get((terrain, swarm_size, method_key), (0.5, 3)),
                    textcoords = "offset points",
                    ha = "center",
                    va = "bottom",
                    fontsize = 7.5,
                    color = color,
                    alpha = 0.9,
                )

        error_axis.set_xticks(x_positions, SWARM_SIZES)
        time_axis.set_xticks(x_positions, SWARM_SIZES)
        if terrain_index == 0:
            error_axis.set_ylabel("Position error (m)")
            time_axis.set_ylabel("Consensus time (s)")

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 0].legend(
        legend_handles,
        legend_labels,
        # Legend placement/styling knobs. Move loc or reduce fontsize if it covers data.
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
    figure.align_ylabels([axes[0, 0], axes[1, 0]])
    figure.suptitle(FIGURE_TITLE, y = 0.98, fontsize = 13)
    # Single shared x-label for the whole figure.
    figure.supxlabel("Swarm size", y = 0.04, fontsize = 11)

    png_path = OUTPUT_STEM + ".png"
    pdf_path = OUTPUT_STEM + ".pdf"
    figure.savefig(png_path, bbox_inches = "tight")
    figure.savefig(pdf_path, bbox_inches = "tight")
    plt.close(figure)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    plot_swarm_size_analysis()
