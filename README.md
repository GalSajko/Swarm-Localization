# Swarm-Localization

Collective localization of an aerial swarm by image retrieval, without GPS. A swarm of agents is dropped over an unknown part of a satellite image, and the swarm must work out where it is by matching what the agents see against a known map.

## Approach

1. **Map codebook (offline).** Each terrain image (city, desert, mountains, rural) is cut into a 20x20 grid of tiles. RootSIFT descriptors of every tile are encoded as a VLAD vector (intra-normalized, then globally L2-normalized) using shared k-means centroids. The 400 tile vectors are the localization hypotheses.
2. **Personal evidence.** Each agent sweeps its own search cell. It collects descriptors during an exploration cycle, builds one VLAD vector from them, and compares it by cosine similarity with all 400 tiles. A softmax with temperature `tau` (`utils.softmax_over_vector`) turns the similarities into a probability vector.
3. **Dempster-Shafer fusion.** The probability vector becomes a mass function. The top-k tiles get singleton masses and spatial neighbor-pair masses (`spatial_topk`). The remaining mass goes to the whole frame Theta, scaled by an entropy-based confidence. Each cycle the agent fuses this with its running belief (**personal fusion**).
4. **Social fusion.** Agents exchange masses with their fixed grid neighbors (ring range 1) and fuse them (**social fusion**). With compatibility boost, neighbors whose hypotheses are the same or adjacent tiles reinforce each other, and incompatible ones are penalized.
5. **Two-stage consensus.** An agent decides ("freezes") once its belief passes a termination rule: its best score passes a fixed or dynamically adapted threshold, its ignorance drops low enough, or enough of its committed mass concentrates on one tile (see `--threshold_type`). Frozen agents then run a confidence-weighted Laplacian average of their position estimates with frozen neighbors until the estimates stabilize.

Results record whether the final tile equals the target tile, the position error in meters, consensus time, and the dispersion of decisions.

## Setup

Run everything from the repository root, because paths are relative. Create a virtual environment at `.venv/`, which the experiment scripts expect, and install the dependencies (tested with Python 3.14). `kagglehub` and `joblib` are only needed for `src/download_dataset.py`, so install them separately if you use that script.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Data files

`data/` and `codebooks/` are git-ignored, so a fresh clone does not contain them. Get the files below from the project owner and copy them into the repository so that it looks like this (`.gitkeep` files are already there):

```
data/
  original_images/
    city_bw.png  desert_bw.png  mountains_bw.png  rural_bw.png   <- grayscale terrain maps
  training_data/
    ims_gray.npy                 <- only needed to retrain the centroids (see below)
codebooks/
  trained_centroids.npy          <- (256, 128) float32 k-means centroids
  vlad_vectors_city.npy  vlad_vectors_desert.npy
  vlad_vectors_mountains.npy  vlad_vectors_rural.npy   <- (400, 256, 128) float32, one row per tile
```

### Prepare the tiles

```bash
.venv/bin/python src/prepare_image_dataset.py
```

This reads `data/original_images/` and `codebooks/trained_centroids.npy` and writes:

- `data/subimages/<terrain>/XX_YY.png`: the 400 tiles of each terrain (20x20 grid). The simulator needs them.
- `codebooks/vlad_vectors_<terrain>.npy`: the VLAD vector of every tile, recomputed from the centroids. With the original centroids the result is identical to the provided files, so the provided `vlad_vectors_*.npy` files are only a backup. Run the script once after copying the files.

### Without `trained_centroids.npy`

If the centroids file is missing, the script trains it first (256 RootSIFT k-means clusters, fixed seed) from `data/training_data/ims_gray.npy` (uint8 array of shape `(1260, 1024, 1024)`; `ims_gray.np` is accepted too), saves it to `codebooks/`, and then continues as above. Training is slow and memory-hungry. `--train` forces retraining even when the centroids exist. Retrained centroids will not equal the paper's, so VLAD vectors and results will differ slightly. To reproduce the paper, use the original `trained_centroids.npy`.

If you do not have `ims_gray.npy`, create it with `.venv/bin/pip install kagglehub joblib` and `.venv/bin/python src/download_dataset.py`. The script downloads the `ims.np` file (about 5 GB, RGBA images) of the Kaggle dataset `sabermalek/spacenetsi` into the kagglehub cache, converts it to grayscale (`cv2.COLOR_RGB2GRAY`) and writes `data/training_data/ims_gray.npy` (1.3 GB). The result is byte-identical to the original `ims_gray.np`. It is not needed to run the simulations. `--force` overwrites an existing file, and the kagglehub cache (`~/.cache/kagglehub`) can be deleted afterwards.

## Running experiments

One experiment is one terrain, one personal fusion rule, one social fusion rule:

```bash
.venv/bin/python src/simulation.py <experiment_name> --<terrain> \
  --threshold_type=dynamic --social_fusion=<rule> --personal_fusion=<rule> [options]
```

- **Terrain:** `--city`, `--desert`, `--mountains` or `--rural` (exactly one).
- **`--threshold_type`:** `dynamic`, `fixed`, `uncertainty` or `concentration` (required). The rule is checked after every decision cycle. It uses the best-tile score `s1` (the mass of the non-Θ focal sets that contain the leading tile, at most 1), the mass on Θ (ignorance) `u`, and the threshold `tau` = `--ds_decision_threshold`. An agent commits to its best tile when:
  - `dynamic`: `s1 >= tau_max - S * (tau_max - 0.15)`, where `tau_max` = `--ds_decision_threshold` and `S` in [0, 1] is the stability of the leading tile. `S` starts at 0 and grows by 0.25 per cycle while the leading tile stays the same, so the threshold falls 1.0, 0.79, 0.58, 0.36, 0.15 over five stable cycles (for `tau_max` = 1). When the leader changes, `S` is scaled by 0.75, 0.5 or 0.25 if the new leader was ranked 2nd, 3rd or 4th in the previous cycle, and reset to 0 otherwise. The useful range of `tau_max` is (0.15, 1].
  - `fixed`: `s1 >= tau`, a constant. With the default `tau` = 1 an agent practically never commits on its own, so give a lower value.
  - `uncertainty`: `u < tau`, regardless of how the committed mass is spread over tiles.
  - `concentration`: `s1 / (1 - u) >= tau`, i.e. the share of the committed (non-Θ) mass that supports the leading tile. Low ignorance alone is not enough if the committed mass is spread, and a high score alone is not enough if much mass is still on Θ.

  `uncertainty` and `concentration` require an explicit `--ds_decision_threshold` in (0, 1]. With every rule an agent can also freeze by adopting the decision of frozen neighbors, and at the time limit every agent takes its current best tile. `--log_signals` adds `u`, the best and second-best scores and the rival score (best score outside the leader's 3x3 neighborhood) to `decision_dynamics.csv`, which helps when tuning `uncertainty` or `concentration`.
- **`--personal_fusion`:** `dempster`, `yager`, `adaptive` (switches between the two when the conflict mass passes 0.5) or `hybrid` (even agent ids use Yager, odd ids use Dempster). Required.
- **`--social_fusion`:** `comp_boost`, `comp_boost_directed` (compatibility only counts for neighbors in the matching direction), `yager`, or `disabled` (no neighbor influence). Required.
- **Other options:** `--trials=`, `--swarm_size=`, `--mass_function=spatial_topk|topk`, `--ds_decision_threshold=`, `--ds_top_k=`, `--ds_max_neighbors=`, `--ds_singleton_weight=`, `--ds_neighbor_compatibility_gain=`, `--ds_neighbor_same_tile_support=`, `--ds_neighbor_adjacent_tile_support=`, `--ds_neighbor_incompatibility_penalty=`. Add `--visualize` to watch a run.

Example, Yager personal fusion with compatibility boost on the desert:

```bash
.venv/bin/python src/simulation.py my_run --desert --threshold_type=dynamic \
  --social_fusion=comp_boost --personal_fusion=yager --trials=200 \
  --ds_neighbor_same_tile_support=0.75 --ds_neighbor_adjacent_tile_support=1.0 \
  --ds_neighbor_incompatibility_penalty=0.1
```

### Predefined sweeps

`experiment_runners/*.sh` run complete sweeps over all four terrains, 200 trials each, and pass every parameter explicitly:

| script | sweeps |
|---|---|
| `personal_fusion_rules_sweep.sh` | dempster, yager, adaptive, hybrid |
| `social_fusion_rules_sweep.sh` | comp_boost, yager, disabled |
| `mass_function_mode_sweep.sh`, `topk_neighbors_sweep.sh`, `singleton_weight_sweep.sh` | mass function, `top_k` and `max_neighbors`, singleton weight |
| `compatibility_gain_sweep.sh`, `incompatibility_penalty_sweep.sh`, `compatibility_gain_incompatibility_penalty_sweep.sh`, `same_and_adjacent_tile_support_sweep.sh` | social fusion parameters |
| `decision_threshold.sh`, `swarm_size.sh` | decision threshold, swarm size |

The scripts open four Terminal windows (macOS `osascript`) and delete the old results of the sweep they run.

### Output

Each experiment appends to four CSVs: `trial_summary.csv`, `agent_final_states.csv`, `decision_dynamics.csv` and `consensus_dispersion.csv`. The folder depends on the experiment name. The sweep names above go to `results/personal_fusion/`, `results/social_fusion/`, `results/decision_threshold/` or `results/swarm_size/`. Any other name goes to `results/continuous_est_resetting_vlad/top_k_ds/<experiment_name>/`.

## Default parameters

Defaults follow Table 1 of the paper. They apply to every run, and the experiment scripts use the same values. A sweep changes only its own parameter.

| parameter | option | default |
|---|---|---|
| personal fusion rule | `--personal_fusion` | `dempster` |
| social fusion rule | `--social_fusion` | `comp_boost` |
| threshold type | `--threshold_type` | `dynamic` |
| initial threshold tau_max | `--ds_decision_threshold` | 1.0 |
| mass function | `--mass_function` | `spatial_topk` |
| top-k hypothesis budget k | `--ds_top_k` | 4 |
| neighbor-retention budget L | `--ds_max_neighbors` | 4 |
| singleton-pair tradeoff gamma | `--ds_singleton_weight` | 0.75 |
| compatibility gain lambda | `--ds_neighbor_compatibility_gain` | 1.0 |
| incompatibility penalty eta | `--ds_neighbor_incompatibility_penalty` | 0.1 |
| same-tile support w_s | `--ds_neighbor_same_tile_support` | 0.75 |
| adjacent-tile support w_a | `--ds_neighbor_adjacent_tile_support` | 1.0 |
| swarm size | `--swarm_size` | 25 |
| trials per setting | `--trials` | 200 |

The paper's default terrain is desert. The terrain flag is always required.

Fixed in code (not CLI options):

| parameter | value | where |
|---|---|---|
| softmax temperature `tau` | 0.02 (the paper's Eq. 6 is a plain softmax, i.e. tau = 1) | `utils.softmax_over_vector` |
| tile grid | 20x20 (400 hypotheses) | `constants.SUBIMAGE_GRID_SIZE` |
| communication ring range | 1 (fixed grid neighbors) | `constants.COMMUNICATION_RING_RANGE` |
| dynamic threshold minimum | 0.15 | `constants.DS_THRESHOLD_MIN` |
| adaptive rule conflict threshold | 0.5 | `decision_making.DS_CONFLICT_MASS_SELECTION_THRESHOLD` |
| simulation speed-up, time limit | 5x, 400 s simulated | `simulation.py` |
