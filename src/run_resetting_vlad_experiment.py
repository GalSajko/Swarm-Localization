#!/usr/bin/env python3
import csv
from pathlib import Path
import shutil

import constants
import decision_making as dm
import simulation


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = REPO_ROOT / constants.CONTINUOUS_RESULTS_DIR / "reseting_vlad.csv"
TERRAINS = constants.TERRAIN_NAMES
EXPERIMENT_NAME = "continuous_resetting_vlad"


def main():
    constants.NUMBER_OF_AGENTS = 9
    dm.configure_dempster_shafer(top_k = 4)
    results_paths = simulation.build_results_paths(EXPERIMENT_NAME)
    experiment_dir = Path(results_paths["trial_summary"]).parent
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    for terrain in TERRAINS:
        simulation.main(
            EXPERIMENT_NAME,
            terrain,
            "yager",
            "dynamic",
            social_fusion_rule = "comp_boost",
            visualize = False,
        )

    trial_summary_path = Path(results_paths["trial_summary"])
    with trial_summary_path.open("r", encoding = "utf-8", newline = "") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        all_rows = list(reader)

    OUTPUT_CSV.parent.mkdir(parents = True, exist_ok = True)
    with OUTPUT_CSV.open("w", encoding = "utf-8", newline = "") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"Wrote comparable trial summary to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
