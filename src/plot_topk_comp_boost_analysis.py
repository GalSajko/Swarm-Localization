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
OUTPUT_STEM = os.path.join(PLOTS_DIR, "topk_comp_boost_analysis")
FIGURE_TITLE = "Effect of Retained Hypothesis Count on Localization Error and Consensus Time"
TERRAINS = ("city", "desert", "mountains", "rural")
TOPK_VALUES = (1, 2, 4, 8)
METHOD_KEY = "comp_boost"
BOX_COLOR = "#4f7ea4"


def load_trial_rows(experiment_name):
    path = os.path.join(RESULTS_ROOT, "top_k_ds", experiment_name, "trial_summary.csv")
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

    for topk in TOPK_VALUES:
        experiment_name = f"topk_{topk}_{METHOD_KEY}"
        rows = load_trial_rows(experiment_name)

        terrain_groups = defaultdict(list)
        for row in rows:
            terrain_groups[row["terrain"]].append(row)

        for terrain in TERRAINS:
            terrain_rows = terrain_groups.get(terrain, [])
            summary[(terrain, "position_error_meters")][topk] = collect_metric_values(
                terrain_rows,
                "position_error_meters",
            )
            summary[(terrain, "consensus_time")][topk] = collect_metric_values(
                terrain_rows,
                "consensus_time",
            )
            timeout_count = sum(1 for row in terrain_rows if parse_bool(row["timeout_reached"]))
            timeout_percentages[terrain][topk] = 100.0 * timeout_count / len(terrain_rows) if terrain_rows else 0.0

    return summary, timeout_percentages


def upper_whisker(values):
    values_array = np.asarray(values, dtype = np.float64)
    q1 = float(np.percentile(values_array, 25))
    q3 = float(np.percentile(values_array, 75))
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    non_outlier_values = values_array[values_array <= upper_threshold]
    return float(np.max(non_outlier_values)) if len(non_outlier_values) else float(np.max(values_array))


def style_boxplot(boxplot, color):
    for box in boxplot["boxes"]:
        box.set(facecolor = color, alpha = 0.35, edgecolor = color, linewidth = 1.4)
    for median in boxplot["medians"]:
        median.set(color = color, linewidth = 2.2)
    for whisker in boxplot["whiskers"]:
        whisker.set(color = color, linewidth = 1.2)
    for cap in boxplot["caps"]:
        cap.set(color = color, linewidth = 1.2)


def plot_topk_comp_boost_analysis():
    os.makedirs(PLOTS_DIR, exist_ok = True)
    summary, timeout_percentages = build_summary()

    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    figure, axes = plt.subplots(2, len(TERRAINS), figsize = (16, 8.5), sharex = True)
    figure.subplots_adjust(left = 0.07, right = 0.995, top = 0.90, bottom = 0.12, wspace = 0.25, hspace = 0.16)

    x_positions = np.arange(1, len(TOPK_VALUES) + 1, dtype = np.float64)
    error_ylim = (0.0, 10000.0)
    time_ylim = (0.0, 420.0)

    for terrain_index, terrain in enumerate(TERRAINS):
        error_axis = axes[0, terrain_index]
        time_axis = axes[1, terrain_index]

        error_axis.set_title(terrain.capitalize())
        error_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)
        time_axis.grid(True, axis = "y", linestyle = "--", linewidth = 0.6, alpha = 0.5)
        error_axis.set_ylim(*error_ylim)
        time_axis.set_ylim(*time_ylim)

        error_values = [summary[(terrain, "position_error_meters")][topk] for topk in TOPK_VALUES]
        time_values = [summary[(terrain, "consensus_time")][topk] for topk in TOPK_VALUES]

        error_boxplot = error_axis.boxplot(
            error_values,
            positions = x_positions,
            widths = 0.28,
            patch_artist = True,
            showfliers = False,
            manage_ticks = False,
        )
        time_boxplot = time_axis.boxplot(
            time_values,
            positions = x_positions,
            widths = 0.28,
            patch_artist = True,
            showfliers = False,
            manage_ticks = False,
        )
        style_boxplot(error_boxplot, BOX_COLOR)
        style_boxplot(time_boxplot, BOX_COLOR)

        for x_coord, values, topk in zip(x_positions, time_values, TOPK_VALUES):
            if not values:
                continue
            time_axis.annotate(
                f"{timeout_percentages[terrain][topk]:.0f}%",
                (x_coord, upper_whisker(values)),
                xytext = (0.5, 3),
                textcoords = "offset points",
                ha = "center",
                va = "bottom",
                fontsize = 7.5,
                color = BOX_COLOR,
                alpha = 0.9,
            )

        error_axis.set_xticks(x_positions, TOPK_VALUES)
        time_axis.set_xticks(x_positions, TOPK_VALUES)
        if terrain_index == 0:
            error_axis.set_ylabel("Position error (m)")
            time_axis.set_ylabel("Consensus time (s)")

    figure.align_ylabels([axes[0, 0], axes[1, 0]])
    figure.suptitle(FIGURE_TITLE, y = 0.98, fontsize = 13)
    figure.supxlabel("Number of retained hypotheses (k)", y = 0.04, fontsize = 11)

    png_path = OUTPUT_STEM + ".png"
    pdf_path = OUTPUT_STEM + ".pdf"
    figure.savefig(png_path, bbox_inches = "tight")
    figure.savefig(pdf_path, bbox_inches = "tight")
    plt.close(figure)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    plot_topk_comp_boost_analysis()
