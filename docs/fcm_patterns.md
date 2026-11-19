# Fuzzy feature pattern localization (`fcm-patterns`)

This document describes how the swarm localizes itself on the `fcm-patterns` branch: how the map is prepared, how each agent searches for and selects a feature, how it obtains its own position, and how the swarm confirms the agents' positions through a fuzzy pattern vote. It is kept up to date with the current implementation.

**Implementation status.** The offline map preparation is implemented in `src/` (see the main README). The online part (search, feature selection, re-centering and pattern voting) has so far been prototyped in simulation scripts that are not part of the repository yet. The values and results below come from those prototypes.

## Contents

1. [Assumptions](#1-assumptions)
2. [Offline: map codebook](#2-offline-map-codebook)
3. [Online overview](#3-online-overview)
4. [Search movement](#4-search-movement)
5. [Feature selection and stop rule](#5-feature-selection-and-stop-rule)
6. [Re-centering and the agent's own position](#6-re-centering-and-the-agents-own-position)
7. [Swarm pattern vote](#7-swarm-pattern-vote)
8. [Final positions](#8-final-positions)
9. [Parameters](#9-parameters)
10. [Measured performance](#10-measured-performance)
11. [Design decisions](#11-design-decisions)
12. [Limitations](#12-limitations)
13. [Open design work](#13-open-design-work)

## 1. Assumptions

- **Map:** one grayscale satellite image per terrain (city, desert, mountains, rural), 8192 × 5837 px, at 2 m per pixel.
- **Agent camera:** looks straight down from a fixed height of 100 m. Its view covers 79 × 63 px (158 × 126 m) of the map.
- **Orientation:** agents are aligned with the map (no rotation).
- **Photo degradation:** every agent photo is a crop of the map image, degraded by `image_augmentation.augment_drone_capture`: a digital zoom of 1.01 to 1.08, resolution loss to 70–85%, and Gaussian noise with sigma 4 to 10, drawn at random for every photo.
- **Relative positions between agents:** measured directly between agents (for example by range and bearing sensing). In the prototype they are exact.
- **No odometry for localization:** an agent's position comes from its match with the map, never from integrating its own motion. Motion knowledge may be used for navigation only, for example to stay inside a search cell.

## 2. Offline: map codebook

Prepared by `src/prepare_image_dataset.py` and described in detail in the main README. In short:

1. **Map features:** SIFT runs on the whole terrain image. For each feature, its pixel position and RootSIFT descriptor are stored in `codebooks/map_features_<terrain>.npz`.
2. **Repeatability filter:** the map is degraded 6 times like an agent photo, and a feature is kept only if it is found again in at least 5 copies (`codebooks/map_repeatability_<terrain>.npz`). This keeps 17–29% of the features: 46,743 in desert up to 385,833 in city.
3. **Fuzzy memberships:** 64 fuzzy c-means centers are trained on RootSIFT descriptors of separate training images (`codebooks/fcm_centers.npz`). Each kept map feature gets its membership in each cluster, with fuzzifier m = 1.1 (`codebooks/fcm_codebook_<terrain>.npz`).

The online stage uses, for the kept features:

| Data | Used for | Source file |
|---|---|---|
| positions | the agent's own position, voting | `fcm_codebook_<terrain>.npz` |
| memberships (K × 64) | pattern vote | `fcm_codebook_<terrain>.npz` |
| centers and membership fuzzifier | memberships of agent features | `fcm_codebook_<terrain>.npz` |
| RootSIFT descriptors (K × 128) | stop rule and the agent's own match | `map_features_<terrain>.npz`, rows with `repeat_counts >= min_repeat_count` |

The codebook does not store descriptors yet, so the prototype takes them from `map_features_<terrain>.npz`, whose kept rows are in the same order as the codebook. Each agent carries 12–99 MB of memberships and 24–197 MB of float32 descriptors (about a quarter of that as 8-bit SIFT values).

## 3. Online overview

1. The swarm is dropped over an unknown part of the map. Each agent gets its own search cell (section 4).
2. Each agent sweeps its cell, taking a photo every 0.5 s, until the strongest feature in a photo is distinctive on the map (section 5).
3. The agent flies until that feature is under the center of its view, takes a new photo there, and takes the matched map feature's position as its own (section 6).
4. The stopped agents vote with the fuzzy memberships of their features and their measured relative positions. The vote confirms the agents' positions and corrects agents with a wrong match (section 7).
5. Every agent's final position is the vote's answer, shifted by its relative position (section 8).

## 4. Search movement

The movement is the grid search of the `main` branch (`simulation.py`, `agent.py`):

- **Search area:** a near-square grid of cells, one per agent (8 agents: 2 × 4, 16 agents: 4 × 4, 25 agents: 5 × 5), centered on the drop point and kept inside the map. A cell is half a tile of the 20 × 20 tile grid: image width / 40 by image height / 40, which is 204.8 × 145.9 px (410 × 292 m).
- **Traversable area:** the cell shrunk by half the view size on each side, so that an agent's view never leaves its cell. Views of different agents therefore never overlap, and no separation rule between agents is needed.
- **Start:** each agent starts at the (min x, min y) corner of its traversable area.
- **Lawnmower sweep:** along a randomly chosen axis. Lanes are 0.8 view heights apart for a horizontal sweep (50.4 px) and 0.8 view widths for a vertical sweep (63.2 px), including the lane on the far border. After the last lane, the sweep goes back.
- **Speed and photos:** 8.5 px (17 m) per simulated second, one photo every 0.5 s, so one photo every 4.25 px (8.5 m). Time limit 400 simulated seconds.

## 5. Feature selection and stop rule

For every photo, the agent:

1. **Detects SIFT keypoints** and takes the one with the highest SIFT response. OpenCV creates a separate keypoint for each dominant orientation at the same spot, so ties at the same position are possible but harmless.
2. **Computes the Lowe ratio** of that feature's RootSIFT descriptor against the map descriptors: d1 / d2, where d1 and d2 are the Euclidean distances to the nearest and the second nearest map descriptor.
3. **Stops if the ratio is below 0.7.** The feature then matches one map feature much better than any other. The nearest map feature is recorded as the agent's match.

An agent that has not stopped within 400 s uses its photo with the lowest Lowe ratio, and its nearest map feature as its match. Such matches are wrong about 3 times in 4.

## 6. Re-centering and the agent's own position

After stopping, the agent does not compute where the feature lies in its photo. Instead:

1. **Centering:** it flies until the selected feature is under the center of its view. The prototype assumes this is precise, for example through image-based steering.
2. **New photo:** it takes a photo at that position. The feature is re-identified in it as the keypoint within 6 px of the view center whose descriptor is closest to the original one. This photo only serves to confirm the centering. The remaining distance between the feature and the view center is about 1.3 m (0.65 px).
3. **Own position:** the agent's position is the map position of the feature it matched in the photo that passed the stop rule.

No odometry and no pixel offset enter this position.

## 7. Swarm pattern vote

The vote uses fuzzy memberships rather than raw descriptors, so that it tolerates appearance changes better than descriptor matching.

**Agent memberships.** Each stopped agent computes the 64 memberships of its selected feature's RootSIFT descriptor with `fuzzy_clustering.calculate_memberships`, using the codebook's centers and membership fuzzifier (1.1).

**Fuzzy overlap.** The similarity between an agent feature u and a map feature v is

```
overlap(u, v) = sum over the 64 clusters of min(u_k, v_k)
```

It is 1 for identical membership vectors and 0 for vectors with no cluster in common. Since both vectors sum to 1, it equals 1 minus half the L1 distance between them. It is the cardinality of the fuzzy intersection (Zadeh), also known as histogram intersection. Unlike cosine similarity, it keeps a partial score when degradation moves the strongest membership to another cluster.

**Candidates.** Every map feature with overlap above 0.5 is a candidate for the agent's feature, weighted by its overlap.

**Votes.** One stopped agent is the reference. If agent i is at relative position r_i from the reference (measured directly), its candidate c votes for the reference being at

```
vote = position of c − r_i
```

**Scoring a place.** All votes within R = 5 px (10 m) of a place are collected. Each agent counts once, with its best overlap among those votes, and the score is the sum over agents. The places scored are the votes that have at least one vote from another agent within R. If no two agents agree anywhere, the single highest-weighted vote is the answer.

**Answer.** The place with the highest score wins. The estimated position of the reference is the overlap-weighted mean of the best vote of each agent agreeing there.

R should match the accuracy of the relative position measurements. With exact measurements, 5 px works better than 15 px, because a wider radius only adds chance agreements between wrong agents.

## 8. Final positions

Each agent's final position is the answer plus its relative position r_i. An agent whose own match (section 6) is more than R from its final position had a wrong match, and its final position comes from the swarm.

The vote outvotes wrong agents because correct agents agree at one place while wrong ones scatter over the map. False places typically gather only 2 or 3 agents by chance.

## 9. Parameters

| Parameter | Value | Section |
|---|---|---|
| meters per pixel | 2.0 | 1 |
| view size | 79 × 63 px | 1 |
| membership fuzzifier | 1.1 | 2, 7 |
| repeatability filter | found in ≥ 5 of 6 degraded copies | 2 |
| cell size | image size / 40 (204.8 × 145.9 px) | 4 |
| lawnmower lane spacing | 0.8 × view height or width | 4 |
| speed | 8.5 px/s (17 m/s) | 4 |
| capture period | 0.5 s | 4 |
| time limit | 400 s | 4, 5 |
| Lowe ratio threshold | 0.7 | 5 |
| re-identification radius | 6 px | 6 |
| candidate overlap threshold | 0.5 | 7 |
| matching radius R | 5 px (10 m) | 7 |

## 10. Measured performance

Prototype results with 100 simulated swarms per terrain and swarm size (8, 16 and 25 agents). Errors are distances between the estimated and the true current position of each agent.

| Terrain | Agents' own match correct (< 10 m) | Swarm found (all agents < 10 m) | Agent error after vote, median | Half the agents done | All agents done, median (8 / 16 / 25) | Agents that never passed the stop rule |
|---|---|---|---|---|---|---|
| city | 100% | 100% | 0.8–0.9 m | 4.0–4.2 s | 13 / 16 / 19 s | 0% |
| rural | 99–100% | 100% | 0.7–0.8 m | 4.8–6.0 s | 18 / 24 / 28 s | 0–1% |
| mountains | 97% | 100% | 0.7–0.8 m | 7.2–7.9 s | 39 / 61 / 231 s | 4–5% |
| desert | 97% | 99–100% | 0.7–0.8 m | 10.8–11.5 s | 44 / 72 / 176 s | 4–5% |

- **Wrong own matches:** among agents that passed the stop rule, 0.1–0.4% of matches are wrong. Among agents that fell back after 400 s, 72–86% are wrong.
- **Waiting time:** in mountains and desert, a few agents find no distinctive feature in their cell. The swarm waits for them although the vote does not need them.
- **Comparison with `main`:** the `main` branch (VLAD tiles with Dempster-Shafer fusion, 25 agents, default settings) reaches the correct tile in 44–72% of trials, with median position errors of 138–379 m, and all agents decide after 50–92 s.

## 11. Design decisions

| Decision | Replaced | Reason |
|---|---|---|
| Lowe ratio < 0.7 as the stop rule | stopping on the first feature found | Map feature density is so high that agents found a matching feature immediately. Agents that pass it have the correct nearest map feature in 99.6–99.9% of cases. |
| Fuzzy overlap with weighted votes | cosine similarity of memberships > 0.9 | Degradation often moves the strongest membership to another cluster. Overlap keeps partial evidence, and the success rate rose from 47% to 69% in single-photo tests. |
| Re-centering on the feature | computing the feature's offset in the photo | The position comes from the map match alone, with no pixel offset calculation. |
| Position from the map match | displacement from the drop point (odometry) | Odometry drifts. The swarm now estimates where its agents are, not where it was dropped. |
| Grid search of `main` | radial search, view-sized jumps | Predictable spacing (270–295 m between neighbours), no overlapping views and no agent wandering off. Localization success was the same with all three movements. |
| Strongest feature per photo | feature nearest the view center | The strongest feature is far more likely to be in the codebook and to survive degradation. |

## 12. Limitations

- **Same image source:** agent photos are degraded crops of the map image. Real photos taken at another time, season or by another camera would match the map far less reliably, especially with raw descriptors (Lowe ratio).
- **Exact measurements:** centering and relative positions between agents are exact in the prototype. Real errors add to the agents' position errors and must stay within R to keep agents agreeing.
- **Central vote:** the vote is computed in one place, instantly, once all agents have stopped. No communication between agents is modeled.
- **No rotation or height change:** agents are aligned with the map at a fixed height.
- **Waiting for the slowest agent:** agents that never pass the stop rule hold the swarm until the time limit.

## 13. Open design work

These ideas are being defined. They are not implemented.

**Looser feature selection with mutual confirmation.** Agents need not select perfect features if the swarm confirms their hypotheses, following the principle of compatibility boost on `main`. Each agent keeps several candidate positions with weights. Neighbours' candidates support each other when they agree with the measured relative position within a tolerance, and candidates nobody supports are penalized. A prototype run with a response-only stop rule (strongest response > 0.063) showed that swarms of 25 agents are still found every time with only 28–49% of agents individually correct, while 8 agents succeed in 60–83% of swarms.

**Evidence accumulation within an agent.** Instead of one decisive feature, an agent combines several consecutive features. Each new feature's candidates support earlier hypotheses that lie at the expected relative position, so wrong hypotheses die out within a few photos. The agent stops when one hypothesis dominates and uses its last photo for the pattern vote. Still to be decided:

- whether the expected relative position comes from the motion between photos or is measured between features visible in the same photo
- how to ensure each photo contributes a new feature, given that consecutive photos overlap by about 95%
- the update rule (additive boost, Bayesian or Dempster-Shafer) and the stop threshold
