# Default parameters
## Personal fusion parameters
- default rule: dempster
- ds_top_k = 4
- ds_max_neighbors = 4
- ds_singleton_weight = 0.75
- mass_function_mode = 'spatial_topk'
## Social fusion parameters
- default rule: comp boost
- ds_neighbor_compatibility_gain = 1.0
- ds_neighbor_same_tile_support = 0.75
- ds_neighbor_adjacent_tile_support = 1.0
- ds_neighbor_incompatibility_penalty = 0.1
## Other parameters
- swarm_size = 25
- simulation_speed_up_factor = 5.0
- threshold type: dynamic
- ds_decision_threshold = 10.0
- terrain: desert
- trials per individual set of parameters: 200

# Experiments plan

## Personal fusion experiments
### 1. set of experiments: personal fusion rules 2D sweep ✅✅
- Changing variable: 
  - personal fusion rule (dempster, yager, adaptive, hybrid)
  - terrain: all 4 terrains
- Results output dir: results/personal_fusion/personal_fusion_rules_sweep
- Plots: 
  - 1st figure:
    - 1st subplot: accuracy (y axis) / personal fusion rule (x axis)
    - 2nd subplot: consensus times (y axis) / personal fusion rule (x axis)
    - four boxplots (one per each terrain) per one x tick
    - both subplots in one figure
  - 2nd figure:
    - 1st subplot: mean pairwise decision distance (y axis) / personal fusion rule (x axis)
    - 2nd subplot: number of connected decision clusters (y axis) / personal fusion rule (x axis)
    - 3rd subplot: largest cluster share (y axis) / personal fusion rule (x axis)
    - four boxplots (one per each terrain) per one x tick
    - all subplots in one figure
    - name of the figure: personal_fusion_rules_sweep_stage1_dispersion
- Expected results:
  - find one best performing personal fusion rule 
  
### 2. set of experiments: ds_top_k, ds_max_neighbor 3D sweep ✅✅
- Changing variables:
  - personal fusion rule: dempster, yager, adaptive, hybrid
  - ds_top_k - (1, 2, 4, 8)
  - ds_max_neighbors - (0, 1, 2, 4, 8)
- Results output dir: results/personal_fusion/topk_neighbors_sweep
- Plots:
  - 1st subplot: accuracy (y axis) / ds_top_k (x axis)
  - 2nd subplot: consensus time (y axis) / ds_top_k (x axis)
  - four boxplots (one per each ds_max_neighbors) per one x tick
  - four plots (2 subplots) in one figure, placed in 2x2 grid
- Expected results:
  - higher value of ds_top_k should lead to better results
  - does increasing ds_max_neighbors value improve the results or just cause additional noise?

### 3. set of experiments: ds_singleton_weight, personal fusion rules 2D sweep ✅✅
- Changing variable: 
  - personal fusion rule: dempster, yager, adaptive, hybrid
  - ds_singleton_weight - (0, 0.25, 0.5, 0.75, 1.0)
- Results output dir: results/personal_fusion/singleton_weight_sweep
- Plots:
  - 1st subplot: accuracy (y axis) / ds_singleton_weight (x axis)
  - 2nd subplot: consensus times (y axis) / ds_singleton_weight (x axis)
  - four boxplots (one per each personal fusion rule) per one x tick
  - both subplots in one figure
- Expected results:
  - higher value of ds_singleton_weight should shorten the consensus time but negatively affect the accuracy - sweet spot should be somewhere in the middle

## Social fusion experiments
### 4. set of experiments: compare social fusion rules 2D sweep ✅✅
- Changing variable: 
  - social fusion rule: yager, comp boost, disabled
  - terrain: all 4 terrains
- Results output dir: results/social_fusion/social_fusion_rules_sweep
- Plots: 
  - 1st figure:
    - 1st subplot: accuracy (y axis) / social fusion rule (x axis)
    - 2nd subplot: consensus times (y axis) / social fusion rule (x axis)
    - one plot (consists of two subplots) per terrain
    - four plots (one per each terrain) in one figure
  - 2nd figure:
    - 1st subplot: mean pairwise decision distance (y axis) / social fusion rule (x axis)
    - 2nd subplot: number of connected decision clusters (y axis) / social fusion rule (x axis)
    - 3rd subplot: largest cluster share (y axis) / social fusion rule (x axis)
    - four boxplots (one per each terrain) per one x tick
    - all subplots in one figure
    - name of the figure: social_fusion_rules_sweep_stage1_dispersion
- Expected results:
  - comp boost and comp boost directed should perform better then yager
  - it should be obvious that disabling social fusion reduces the performance 
  
### 5. set of experiments: ds_neighbor_compatibility_gain parameter 1D sweep ✅✅
- Changing variable: 
  - ds_neighbor_compatibility_gain - (0, 0.25, 0.5, 0.75, 1.0)
- Results output dir: results/social_fusion/compatibility_gain_sweep
- Plots:
  - 1st subplot: accuracy (y axis) / ds_neighbor_compatibility_gain (x axis)
  - 2nd subplot: consensus times (y axis) / ds_neighbor_compatibility_gain (x axis)
- Expected results:
  - higher value should improve the results

#### 5.1 set of experiments: ds_neighbor_compatibility_gain, ds_neighbor_incompatibility_penalty 2D sweep ✅✅
- Changing variables:
  - ds_neighbor_compatibility_gain - (0, 0.25, 0.5, 0.75, 1.0)
  - ds_neighbor_incompatibility_penalty - (0, 0.25, 0.5, 0.75, 1.0)
- Results output dir: results/social_fusion/compatibility_gain_incompatibility_penalty_sweep
- Plot:
  - 1st subplot: accuracy (y axis) / ds_neighbor_compatibility_gain (x axis)
  - 2nd subplot: consensus time (y axis) / ds_neighbor_compatibility_gain (x axis)
  - four boxplots (one per each incompatibility penalty value) per one x tick

### 6. set of experiments: ds_neighbor_same_tile_support, ds_neighbor_adjacent_tile_support 2D sweep ✅
- Changing variables:
  - ds_neighbor_same_tile_support - (0, 0.25, 0.5, 0.75, 1.0)
  - ds_neighbor_adjacent_tile_support - (0, 0.25, 0.5, 0.75, 1.0)
- Results output dir: results/social_fusion/same_and_adjacent_tile_support_sweep
- Plots:
  - 1st subplot: accuracy (y axis) / ds_neighbor_same_tile_support (x axis)
  - 2nd subplot: consensus times (y axis) / ds_neighbor_same_tile_support (x axis)
  - five boxplots (one per each ds_neighbor_adjacent_tile_support value) per one x tick
  - both subplots in one figure
- Expected results:
  - values in the middle should bring the best results

### 7. set of experiments: ds_neighbor_incompatibility_penalty 1D sweep ✅
- Changing variable: 
  - ds_neighbor_incompatibility_penalty - (0, 0.25, 0.5, 0.75, 1.0)
- Results output dir: results/social_fusion/incompatibility_penalty_sweep
- Plots:
  - 1st subplot: accuracy (y axis) / ds_neighbor_incompatibility_penalty (x axis)
  - 2nd subplot: consensus time (y axis) / ds_neighbor_incompatibility_penalty (x axis)
  - both subplots in one figure
- Expected results: 
  - higher penalty should improve the results

## Decision threshold experiments
### 8. set of experiments: decision threshold type 3D sweep
- Changing variables: 
  - decision threshold type (fixed, dynamic)
  - ds_decision_threshold - (0.5, 5, 10, 15)
  - terrain: all 4 terrains
- Results output dir: results/decision_threshold
- Plots:
  - 1st figure:
    - 1st subplot: accuracy (y axis) / ds_decision_threshold (x axis)
    - 2nd subplot: consensus time (y axis) / ds_decision_threshold (x axis)
    - two boxplots (one per each decision threshold type) per one x tick
    - four plots (one per each terrain) in one figure
  - 2nd figure:
    - dynamic decision thresholds values of all agents (y axis) / time
    - only for one trial
    - use line plot, not default boxplots

## Swarm size experiments ✅✅
### 9. set of experiments: swarm size 2D sweep
- Changing variables: 
  - swarm size - (2, 16, 36, 64, 100)
  - terrain: all 4 terrains
- Results output dir: results/swarm_size
- Plots:
  - 1st subplot: accuracy (y axis) / swarm_size (x axis)
  - 2nd subplot: consensus time (y axis) / swarm_size (x axis)
  - four boxplots (one per each terrain) per one x tick
  - both subplots in one figure

## Other implementation rules
- each experiment (batch of default number of trials per one set of parameters) should run in its own terminal window
- all sets of experiments can run 4 experiments in parallel (each in its own terminal window)
- if results output directory does not exist, create it

## Plotting rules
- default plot stye: boxplot with median and whiskers, following Tukey's convention
- when plot consists of multiple subplots always place them one above another
- when figure consists of four plots, place them in 2x2 grid
- when subplots share same x axis, write x label and x ticks values only on the lowest subplot's x axis
- accuracy axis label: "position error / m"
- consensus time axis label: "consensus time / s"

## Results metrics
- accuracy is defined as median of per-trial position errors in meters
- consensus time is defined as median of per-trial consensus times
- timeout percentage is defined as percentage of trials that finished due to timeout
- timeout trials should not be included in accuracy and consensus time calculations
- timeout percentage should be presented as a number above whiskers on consensus time plots