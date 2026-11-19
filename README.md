# Swarm-Localization

Collective localization of an aerial swarm by image retrieval, without GPS. A swarm of agents is dropped over an unknown part of a satellite image, and the swarm must work out where it is by matching what the agents see against a known map.

## Approach

This branch (`fcm-patterns`) replaces tile classification with fuzzy feature pattern matching. The map is no longer cut into tiles and described with VLAD vectors. Instead, the swarm matches the pattern of features that its agents detect against the features of the whole map.

1. **Map codebook (offline, implemented).** RootSIFT features are calculated on each whole terrain image (city, desert, mountains, rural) and saved with their pixel locations. A shared set of 64 fuzzy c-means centers is trained on the SpaceNet training images. Only the repeatable map features are kept, i.e. those that are found again when the map is degraded like an agent's photo, and each gets a membership in each of the 64 clusters. See [Prepare the codebooks](#prepare-the-codebooks).
2. **Pattern matching (online, not implemented yet).** Each agent detects a feature and calculates its memberships with the same centers and the same fuzzifier (`fuzzy_clustering.calculate_memberships`). The swarm then looks for map features with similar memberships and the same relative positions. For now a feature is assumed to lie in the middle of the agent's view, and agents are assumed to be aligned with the map (no rotation). The current design and its prototype results are described in [docs/fcm_patterns.md](docs/fcm_patterns.md).

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
.venv/bin/python src/prepare_image_dataset.py [--train] [--extract] [--repeatability] [--training_m=<m>] [--membership_m=<m>] [--min_repeat_count=<k>]
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

**Step 3** counts in how many degraded copies of the terrain image every map feature is found again (see [Repeatability filter](#repeatability-filter)). `map_repeatability_<terrain>.npz` holds `repeat_counts` `(N,)` and `number_of_copies`.

**Step 4** keeps the repeatable map features and calculates their membership in every cluster. `fcm_codebook_<terrain>.npz` is the only file the online stage needs. With K kept features:

- `keypoints_xy` `(K, 2)` and `map_shape`: as above.
- `memberships` `(K, 64)`: each row sums to 1.
- `repeat_counts` `(K,)`: in how many degraded copies each kept feature was found.
- `centers` `(64, 128)`: agents use them to calculate memberships of the features they detect.
- `membership_fuzzifier`: agents must use the same value. `training_fuzzifier` is saved for reference.
- `min_repeat_count` and `number_of_degraded_copies`: the filter settings the codebook was made with.

#### Repeatability filter

An agent's photo is degraded (zoom, resolution loss, noise; `image_augmentation.augment_drone_capture`), and most map features are not detected again in such a photo. The codebook therefore keeps only the features that survive this degradation.

1. The whole terrain image is degraded `constants.NUMBER_OF_DEGRADED_COPIES` (6) times with the same three operations (`image_augmentation.degrade_whole_image`). The strengths are spread evenly over the ranges used for agents' photos (zoom 1.01 to 1.08, resolution 70% to 85%, noise sigma 4 to 10), in a different combination for every copy.
2. SIFT runs on every copy, and the keypoints are converted back to the pixels of the original image.
3. A map feature is found in a copy if the copy has a keypoint within 2.5 pixels, with a size within a factor of 1.3 and an orientation within 20 degrees (`feature_repeatability.py`).
4. A feature is kept if it is found in at least `constants.MIN_REPEAT_COUNT` (`--min_repeat_count`, default 5) copies. `--min_repeat_count=0` keeps all features.

Agents cannot calculate repeatability, and they do not filter the features they detect. Repeatable features tend to be larger and to have a higher SIFT response, so an agent should prefer its features with the highest response.

Changing the required number of copies only repeats step 4:

```bash
.venv/bin/python src/prepare_image_dataset.py --min_repeat_count=4
```

#### The two fuzzifiers

The membership of a descriptor in cluster k is `d_k^(-2/(m-1)) / sum_j d_j^(-2/(m-1))`, where `d_k` is its distance to center k and `m > 1` is the fuzzifier. Memberships approach a hard (k-means) assignment as m approaches 1, and become uniform as m grows.

- **Training fuzzifier** (`constants.TRAINING_FUZZIFIER`, `--training_m`, default 1.1) shapes the centers. Changing it requires `--train`.
- **Membership fuzzifier** (`constants.MEMBERSHIP_FUZZIFIER`, `--membership_m`, default 1.1) converts distances to memberships for the map and for the agents. Changing it only repeats step 4:

```bash
.venv/bin/python src/prepare_image_dataset.py --membership_m=1.2
```
