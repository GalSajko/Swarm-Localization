# Swarm-Localization

Collective localization of an aerial swarm by image retrieval, without GPS. A swarm of agents is dropped over an unknown part of a satellite image, and the swarm must work out where it is by matching what the agents see against a known map.

## Approach

This branch (`fcm-patterns`) replaces tile classification with fuzzy feature pattern matching. The map is no longer cut into tiles and described with VLAD vectors. Instead, the swarm matches the pattern of features that its agents detect against the features of the whole map.

1. **Map codebook (offline, implemented).** RootSIFT features are calculated on each whole terrain image (city, desert, mountains, rural) and saved with their pixel locations. A shared set of 64 fuzzy c-means centers is trained on the SpaceNet training images. A window of the agent's view size slides over the map, and only the most repeatable feature of every window is kept (repeatable features are found again when the map is degraded like an agent's photo). This leaves about 3 features per agent view on every terrain. Each kept feature gets a membership in each of the 64 clusters. See [Prepare the codebooks](#prepare-the-codebooks).
2. **Pattern matching (online, implemented as a simulation).** Each agent sweeps its own search cell and stops when the strongest feature in its photo passes a stop rule. It flies until that feature is under the center of its view, and takes the map position of the map feature its descriptor matched as its own position, without odometry. The agents then vote with the fuzzy memberships of their features (calculated with the same centers and fuzzifier, `fuzzy_clustering.calculate_memberships`) and their measured relative positions, which confirms correct agents and corrects wrong ones. Agents are assumed to be aligned with the map up to a small heading error (at most 5 degrees) in their photos. See [Run the online simulation](#run-the-online-simulation).

The previous approach (VLAD tile classification with Dempster-Shafer fusion) and its simulator live on the `main` branch.

## Setup

Run everything from the repository root, because paths are relative. Create a virtual environment at `.venv/` and install the dependencies (tested with Python 3.14). `kagglehub` and `joblib` are only needed for `src/download_dataset.py`, so install them separately if you use that script.

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
    ims_gray.npy                 <- only needed to train the centers (see below)
```

If you do not have `ims_gray.npy`, create it with `.venv/bin/pip install kagglehub joblib` and `.venv/bin/python src/download_dataset.py`. The script downloads the `ims.np` file (about 5 GB, RGBA images) of the Kaggle dataset `sabermalek/spacenetsi` and writes the grayscale array of shape `(1260, 1024, 1024)` to `data/training_data/ims_gray.npy` (`ims_gray.np` is accepted too).

### Prepare the codebooks

```bash
.venv/bin/python src/prepare_image_dataset.py [--train] [--extract] [--repeatability] [--training_m=<m>] [--membership_m=<m>] [--window_steps=<s>]
```

The script has four steps. The first three are slow and only run when their output is missing or when forced. The fourth is fast and always runs.

| step | output in `codebooks/` | runs when |
|---|---|---|
| 1. Train the centers | `fcm_centers.npz` | file is missing, or `--train` |
| 2. Calculate map features | `map_features_<terrain>.npz` | file is missing, or `--extract` |
| 3. Calculate repeatability | `map_repeatability_<terrain>.npz` | file is missing or does not fit the map features, or `--extract` or `--repeatability` |
| 4. Select features and calculate memberships | `fcm_codebook_<terrain>.npz` | always |

**Step 1** samples up to 1000 RootSIFT descriptors from each training image, initializes 64 centers with k-means on 200,000 of them (fixed seed), and refines the centers with fuzzy c-means. `fcm_centers.npz` holds `centers` `(64, 128)` and `training_fuzzifier`.

**Step 2** runs SIFT on the whole terrain image. `map_features_<terrain>.npz` holds, for the N features of the map:

- `keypoints_xy` `(N, 2)`: pixel location (x, y) in the terrain image.
- `keypoints_meta` `(N, 3)`: SIFT size, angle and response.
- `descriptors` `(N, 128)`: RootSIFT descriptors. They are kept so that step 4 can be repeated without step 2.
- `map_shape`: (height, width) of the terrain image.

**Step 3** counts in how many degraded copies of the terrain image every map feature is found again (see [Repeatability](#repeatability)). `map_repeatability_<terrain>.npz` holds `repeat_counts` `(N,)` and `number_of_copies`.

**Step 4** selects the codebook features with a sliding window (see [Window selection](#window-selection)) and calculates their membership in every cluster. `fcm_codebook_<terrain>.npz` is the only file the online stage needs (besides the terrain image, which the simulation uses to take the agents' photos). Codebooks made with an older version of this step must be prepared again, which only repeats this step. With K kept features:

- `keypoints_xy` `(K, 2)` and `map_shape`: as above.
- `descriptors` `(K, 128)`: RootSIFT descriptors. Agents match their own descriptors against them.
- `memberships` `(K, 64)`: each row sums to 1.
- `repeat_counts` `(K,)`: in how many degraded copies each kept feature was found.
- `centers` `(64, 128)`: agents use them to calculate memberships of the features they detect.
- `membership_fuzzifier`: agents must use the same value. `training_fuzzifier` is saved for reference.
- `window_steps` and `number_of_degraded_copies`: the selection settings the codebook was made with.

#### Repeatability

An agent's photo differs from the map (see [Agent photos](#agent-photos)), and most map features are not detected again in it. Step 3 therefore measures how well every map feature survives a mild version of this degradation (zoom, resolution loss, noise).

1. The whole terrain image is degraded `constants.NUMBER_OF_DEGRADED_COPIES` (6) times with the same three operations (`image_augmentation.degrade_whole_image`). The strengths are spread evenly over the ranges (zoom 1.01 to 1.08, resolution 70% to 85%, noise sigma 4 to 10), in a different combination for every copy.
2. SIFT runs on every copy, and the keypoints are converted back to the pixels of the original image.
3. A map feature is found in a copy if the copy has a keypoint within 2.5 pixels, with a size within a factor of 1.3 and an orientation within 20 degrees (`feature_repeatability.py`).
4. The repeat count of a feature is the number of copies in which it is found (0 to 6).

Agents cannot calculate repeatability, and they do not filter the features they detect. Repeatable features tend to be larger and to have a higher SIFT response, so an agent should prefer its features with the highest response.

#### Window selection

Repeatable features are much denser in some terrains than in others (a view in city contains about 40 features found in at least 5 of 6 copies, in desert about 5). Step 4 evens out the density:

1. A window of the agent's view size (79 x 63 pixels) slides over the whole map with a step of 1/`constants.WINDOW_STEPS` (5) of its size in each direction.
2. In every window the feature with the highest repeat count wins. Ties are broken by the highest SIFT response.
3. The codebook keeps every feature that wins at least one window. Overlapping windows often share the winner, so this leaves about 27,000 to 31,000 features per terrain, about 3 per agent view. Only windows without any map feature (up to 6% in mountains) stay empty.

Changing the window step only repeats step 4:

```bash
.venv/bin/python src/prepare_image_dataset.py --window_steps=4
```

#### The two fuzzifiers

The membership of a descriptor in cluster k is `d_k^(-2/(m-1)) / sum_j d_j^(-2/(m-1))`, where `d_k` is its distance to center k and `m > 1` is the fuzzifier. Memberships approach a hard (k-means) assignment as m approaches 1, and become uniform as m grows.

- **Training fuzzifier** (`constants.TRAINING_FUZZIFIER`, `--training_m`, default 1.1) shapes the centers. Changing it requires `--train`.
- **Membership fuzzifier** (`constants.MEMBERSHIP_FUZZIFIER`, `--membership_m`, default 1.1) converts distances to memberships for the map and for the agents. Changing it only repeats step 4:

```bash
.venv/bin/python src/prepare_image_dataset.py --membership_m=1.2
```

## Run the online simulation

```bash
.venv/bin/python src/swarm_simulation.py [--experiment_name=<name>] [--terrains <terrain> ...] [--number_of_agents=<n>] [--trials=<t>] [--stop_rule=response|lowe] [--stop_threshold=<value>] [--seed=<s>]
```

Defaults: experiment `default`, all four terrains, 25 agents, 100 trials per terrain, stop rule `response`, seed 0 (trial i uses seed + i). The swarm size must be an even number or a perfect square, as on the `main` branch.

### What a trial simulates

1. **Search area.** A near-square grid of cells, one per agent, is centered on a random drop point inside the map. A cell is half a tile of a 20x20 tile grid (about 410 x 290 m). This is the grid search of the `main` branch.
2. **Search.** Every agent sweeps its cell in lawnmower lanes 0.8 view sizes apart, at 17 m/s, and takes a photo every 0.5 s (`search_agent.py`). The photo differs from the map like a drone photo from a satellite image (see [Agent photos](#agent-photos)). In every photo the agent takes its strongest feature (highest SIFT response) and stops when it passes the stop rule:
   - `response`: the response is above `constants.RESPONSE_THRESHOLD` (0.0633, the 90th percentile of the strongest response in random photos, pooled over the four terrains, measured with the earlier mild photo degradation; the current photos pass it more often).
   - `lowe`: the Lowe ratio of its RootSIFT descriptor against the codebook descriptors (distance to the nearest divided by the distance to the second nearest) is below `constants.LOWE_RATIO_THRESHOLD` (0.7).

   An agent that does not stop within 400 s uses its best photo (highest response, or lowest Lowe ratio).
3. **Re-centering.** The agent flies until the selected feature is under the center of its view and looks again. Its own position estimate is the map position of the codebook feature that its descriptor matched best. No odometry and no offset within the photo are used.
4. **Pattern vote** (`pattern_voting.py`). Every agent calculates the memberships of its feature. Its candidates are the map features whose fuzzy overlap `sum_k min(u_k, v_k)` with these memberships exceeds 0.5. Each candidate votes for the position of a reference agent, using the agents' relative positions, which are measured directly between the agents (exact in the simulation). Votes of different agents agree within 5 px (10 m). A place is scored by the sum over agents of each agent's best overlap there, and the best place is the answer. Every agent's final estimate is the answer shifted by its relative position.

The simulation knows the true positions only to take photos and to emulate flying. Agents never use them.

### Agent photos

A drone photo differs from the satellite map in geometry, light and sensor. `image_augmentation.degrade_drone_photo` simulates this for every photo, with random strengths in the ranges of the `PHOTO_*` constants:

| difference | range |
|---|---|
| scale (altitude error, different ground resolution) | 0.85 to 1.20 |
| heading error | up to 5 degrees |
| camera tilt (perspective) | every corner moves by up to 3% of the view size |
| view center shift | up to 1 px |
| gamma, contrast, brightness | 0.7 to 1.4, 0.75 to 1.3, up to 20 gray levels |
| illumination gradient across the view (sun angle, vignetting, cloud shadow) | up to 15% |
| blur (defocus, motion) | sigma 0.4 to 1.2 |
| resolution | 50% to 80% |
| noise | sigma 5 to 14 |
| JPEG compression | quality 40 to 80 |

The geometric part is one homography, which also gives the true map position of every keypoint in the photo (used only by the simulation). Seasonal changes, shadows at another time of day, changed buildings and parallax of tall objects cannot be simulated by filtering the map image, so real image pairs remain the decisive test.

### Results

Results are appended to two csv files in `results/fcm_patterns/<experiment_name>/`:

- `trial_summary.csv`: one row per trial. `swarm_correct` is True if every agent's final estimate is within 10 m (`constants.CORRECT_POSITION_TOLERANCE`) of its true position. It also records the median and maximum agent errors, the share of agents whose own match was correct, how many agents voted, agreed, fell back or never detected a feature, the first, half (ceil(n/2)-th) and last decision time of the agents (stop time plus the re-centering flight), whether any agent hit the 400 s limit, and the median centering residual.
- `agent_final_states.csv`: one row per agent, with its stop and decision time, the response and Lowe ratio of its selected feature, its true position, its own and final estimates with their errors, and whether it agreed with the swarm.

Errors are distances between estimated and true current agent positions, in meters (2 m per pixel).
