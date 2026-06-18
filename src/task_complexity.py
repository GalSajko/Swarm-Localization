import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import constants
import utils

OUTPUT_DIR = "results/task_complexity"
LOW_FEATURE_THRESHOLD = 20
TERRAIN_PLOT_COLORS = {
    "city": "#90a4ae",
    "desert": "#d7c3a3",
    "mountains": "#aab7a2",
    "rural": "#b8cbb8",
}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_subimage_indices(filename):
    stem = os.path.splitext(filename)[0]
    x_str, y_str = stem.split("_")
    return int(x_str), int(y_str)


def list_available_terrains():
    terrains = []
    if not os.path.isdir(constants.SUBIMAGES_DIR):
        return terrains

    for entry in sorted(os.listdir(constants.SUBIMAGES_DIR)):
        terrain_dir = os.path.join(constants.SUBIMAGES_DIR, entry)
        if os.path.isdir(terrain_dir):
            terrains.append(entry)

    return terrains


def load_terrain_subimages(terrain_name):
    terrain_dir = os.path.join(constants.SUBIMAGES_DIR, terrain_name)
    filenames = sorted([f for f in os.listdir(terrain_dir) if f.lower().endswith(".png")], key=parse_subimage_indices)

    images = []
    coordinates = []
    for filename in filenames:
        image = cv2.imread(os.path.join(terrain_dir, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        images.append(image)
        coordinates.append(parse_subimage_indices(filename))

    return filenames, images, coordinates


def infer_grid_shape(coordinates):
    max_x = max(x for x, _ in coordinates)
    max_y = max(y for _, y in coordinates)
    return max_y, max_x


def compute_root_sift_keypoint_counts(images):
    sift = constants.SIFT_EXTRACTOR
    keypoint_counts = []

    for image in images:
        _, descriptors = sift.detectAndCompute(image, None)
        root_sift_descriptors = utils.root_sift(descriptors)
        if root_sift_descriptors is None:
            keypoint_counts.append(0)
        else:
            keypoint_counts.append(len(root_sift_descriptors))

    return np.asarray(keypoint_counts, dtype=np.float32)


def load_terrain_vlad_vectors(terrain_name):
    vlad_path = os.path.join(constants.CODEBOOKS_DIR, f"vlad_vectors_{terrain_name}.npy")
    return np.load(vlad_path)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def compute_similarity_matrix(vlad_vectors):
    flattened = vlad_vectors.reshape(vlad_vectors.shape[0], -1).astype(np.float32)
    normalized = normalize_rows(flattened)
    return normalized @ normalized.T


def compute_nearest_nonself_similarity(similarity_matrix):
    masked = similarity_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    return np.max(masked, axis=1)


def compute_neighbor_similarity(coordinates, similarity_matrix):
    index_by_coordinate = {coordinate: index for index, coordinate in enumerate(coordinates)}
    neighbor_similarity = np.zeros(len(coordinates), dtype=np.float32)

    for index, (x, y) in enumerate(coordinates):
        similarities = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor_coordinate = (x + dx, y + dy)
                if neighbor_coordinate in index_by_coordinate:
                    neighbor_index = index_by_coordinate[neighbor_coordinate]
                    similarities.append(similarity_matrix[index, neighbor_index])

        neighbor_similarity[index] = float(np.mean(similarities)) if similarities else 0.0

    return neighbor_similarity


def min_max_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    value_min = float(np.min(values))
    value_max = float(np.max(values))

    if value_max - value_min < 1e-12:
        return np.zeros_like(values, dtype=np.float32)

    return (values - value_min) / (value_max - value_min)


def build_monotony_heatmap(coordinates, grid_shape, nearest_similarity, neighbor_similarity, keypoint_counts):
    nearest_norm = min_max_normalize(nearest_similarity)
    neighbor_norm = min_max_normalize(neighbor_similarity)
    texture_poverty = 1.0 - min_max_normalize(keypoint_counts)

    monotony_score = 0.45 * nearest_norm + 0.40 * neighbor_norm + 0.15 * texture_poverty
    heatmap = np.full(grid_shape, np.nan, dtype=np.float32)

    for index, (x, y) in enumerate(coordinates):
        heatmap[y - 1, x - 1] = monotony_score[index]

    return heatmap, monotony_score


def compute_fraction_below_threshold(values, threshold):
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(values < threshold))


def save_heatmap(terrain_name, heatmap):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    im = plt.imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
    plt.colorbar(im, label="Monotonousness")
    plt.title(f"{terrain_name.capitalize()} Terrain Monotony Heatmap")
    plt.xlabel("Subimage column")
    plt.ylabel("Subimage row")
    plt.xticks(np.arange(heatmap.shape[1]))
    plt.yticks(np.arange(heatmap.shape[0]))
    plt.grid(color="white", linestyle=":", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{terrain_name}_heatmap.png"))
    plt.close()


def create_boxplot(metric_values_by_terrain, ylabel, title, output_filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metric_values_by_terrain = sorted(
        metric_values_by_terrain,
        key=lambda item: float(np.median(item[1])),
    )
    terrain_names = [terrain_name for terrain_name, _ in metric_values_by_terrain]
    metric_values = [values for _, values in metric_values_by_terrain]
    colors = [TERRAIN_PLOT_COLORS.get(terrain_name, "#b0bec5") for terrain_name in terrain_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot = ax.boxplot(
        metric_values,
        patch_artist=True,
        whis=1.5,
        tick_labels=[terrain_name.capitalize() for terrain_name in terrain_names],
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "#5f6b73",
            "markeredgecolor": "#5f6b73",
            "alpha": 0.55,
        },
        medianprops={"color": "#37474f", "linewidth": 1.5},
        whiskerprops={"color": "#78909c", "linewidth": 1.0},
        capprops={"color": "#78909c", "linewidth": 1.0},
        boxprops={"edgecolor": "#78909c", "linewidth": 1.0},
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Terrain")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cfd8dc")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close(fig)


def compute_raw_terrain_metrics(terrain_name):
    filenames, images, coordinates = load_terrain_subimages(terrain_name)
    if not filenames:
        raise ValueError(f"No subimages found for terrain '{terrain_name}'.")

    grid_shape = infer_grid_shape(coordinates)
    vlad_vectors = load_terrain_vlad_vectors(terrain_name)
    keypoint_counts = compute_root_sift_keypoint_counts(images)
    similarity_matrix = compute_similarity_matrix(vlad_vectors)

    nearest_similarity = compute_nearest_nonself_similarity(similarity_matrix)
    neighbor_similarity = compute_neighbor_similarity(coordinates, similarity_matrix)
    heatmap, monotony_score = build_monotony_heatmap(
        coordinates,
        grid_shape,
        nearest_similarity,
        neighbor_similarity,
        keypoint_counts,
    )

    metrics = {
        "terrain": terrain_name,
        "num_subimages": len(filenames),
        "grid_rows": grid_shape[0],
        "grid_cols": grid_shape[1],
        "mean_nearest_nonself_similarity": float(np.mean(nearest_similarity)),
        "median_nearest_nonself_similarity": float(np.median(nearest_similarity)),
        "mean_neighbor_similarity": float(np.mean(neighbor_similarity)),
        "mean_keypoint_count": float(np.mean(keypoint_counts)),
        "median_keypoint_count": float(np.median(keypoint_counts)),
        "zero_feature_fraction": compute_fraction_below_threshold(keypoint_counts, 1),
        "low_feature_fraction": compute_fraction_below_threshold(keypoint_counts, LOW_FEATURE_THRESHOLD),
        "mean_monotony_score": float(np.nanmean(heatmap)),
        "heatmap": heatmap,
        "monotony_score": monotony_score,
        "nearest_similarity_values": nearest_similarity,
        "keypoint_count_values": keypoint_counts,
    }

    return metrics


def compute_complexity_scores(metrics_per_terrain):
    nearest_values = np.array([m["mean_nearest_nonself_similarity"] for m in metrics_per_terrain], dtype=np.float32)
    neighbor_values = np.array([m["mean_neighbor_similarity"] for m in metrics_per_terrain], dtype=np.float32)
    keypoint_values = np.array([m["mean_keypoint_count"] for m in metrics_per_terrain], dtype=np.float32)

    repetitiveness = min_max_normalize(nearest_values)
    local_ambiguity = min_max_normalize(neighbor_values)
    texture_poverty = 1.0 - min_max_normalize(keypoint_values)

    for index, metrics in enumerate(metrics_per_terrain):
        complexity_score = (
            0.50 * repetitiveness[index]
            + 0.30 * local_ambiguity[index]
            + 0.20 * texture_poverty[index]
        )
        metrics["task_complexity"] = float(complexity_score)
        metrics["repetitiveness_component"] = float(repetitiveness[index])
        metrics["local_ambiguity_component"] = float(local_ambiguity[index])
        metrics["texture_poverty_component"] = float(texture_poverty[index])


def save_summary(metrics_per_terrain):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "terrain_complexity_summary.csv")

    fieldnames = [
        "terrain",
        "num_subimages",
        "grid_rows",
        "grid_cols",
        "mean_nearest_nonself_similarity",
        "median_nearest_nonself_similarity",
        "mean_neighbor_similarity",
        "mean_keypoint_count",
        "median_keypoint_count",
        "zero_feature_fraction",
        "low_feature_fraction",
        "mean_monotony_score",
        "repetitiveness_component",
        "local_ambiguity_component",
        "texture_poverty_component",
        "task_complexity",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in sorted(metrics_per_terrain, key=lambda item: item["task_complexity"], reverse=True):
            writer.writerow({field: metrics[field] for field in fieldnames})


def main():
    terrains = list_available_terrains()
    if not terrains:
        raise ValueError("No terrain subimage directories found.")

    metrics_per_terrain = [compute_raw_terrain_metrics(terrain) for terrain in terrains]
    compute_complexity_scores(metrics_per_terrain)

    for metrics in metrics_per_terrain:
        save_heatmap(metrics["terrain"], metrics["heatmap"])

    create_boxplot(
        [
            (metrics["terrain"], metrics["nearest_similarity_values"])
            for metrics in metrics_per_terrain
        ],
        ylabel="Nearest Non-self VLAD Cosine Similarity",
        title="Nearest Non-self Similarity by Terrain",
        output_filename="nearest_nonself_similarity_boxplot.png",
    )
    create_boxplot(
        [
            (metrics["terrain"], metrics["keypoint_count_values"])
            for metrics in metrics_per_terrain
        ],
        ylabel="RootSIFT Feature Count per Subimage",
        title=f"Feature Count by Terrain (low-feature threshold = {LOW_FEATURE_THRESHOLD})",
        output_filename="feature_count_boxplot.png",
    )

    save_summary(metrics_per_terrain)

    for metrics in sorted(metrics_per_terrain, key=lambda item: item["task_complexity"], reverse=True):
        print(
            f"Terrain: {metrics['terrain']}\n"
            f"  Subimages analyzed: {metrics['num_subimages']} ({metrics['grid_rows']}x{metrics['grid_cols']} grid)\n"
            f"  Median nearest non-self VLAD cosine similarity: {metrics['median_nearest_nonself_similarity']:.3f}\n"
            f"  Mean nearest non-self VLAD cosine similarity: {metrics['mean_nearest_nonself_similarity']:.3f}\n"
            f"  Median RootSIFT feature count per subimage: {metrics['median_keypoint_count']:.1f}\n"
            f"  Mean RootSIFT feature count per subimage: {metrics['mean_keypoint_count']:.1f}\n"
            f"  Fraction of zero-feature tiles: {metrics['zero_feature_fraction']:.3f} ({metrics['zero_feature_fraction'] * 100:.1f}%)\n"
            f"  Fraction of low-feature tiles (< {LOW_FEATURE_THRESHOLD} features): "
            f"{metrics['low_feature_fraction']:.3f} ({metrics['low_feature_fraction'] * 100:.1f}%)\n"
        )


if __name__ == "__main__":
    main()
