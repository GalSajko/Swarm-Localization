import numpy as np
from math import comb

DS_TOP_K = 4
DS_SINGLETON_WEIGHT = 0.75
DS_MAX_NEIGHBORS = 4
DS_MASS_FUNCTION_MODE = "spatial_topk"
DS_NEIGHBOR_COMPATIBILITY_GAIN = 1.0
DS_NEIGHBOR_SAME_TILE_SUPPORT = 0.5
DS_NEIGHBOR_ADJACENT_TILE_SUPPORT = 0.5
DS_NEIGHBOR_INCOMPATIBILITY_PENALTY = 0.5

# Applied only when using the "adaptive" personal fusion rule to select between Dempster's and Yager's rules.
DS_CONFLICT_MASS_SELECTION_THRESHOLD = 0.5

def configure_dempster_shafer(
    top_k = None,
    max_neighbors = None,
    mass_function_mode = None,
    singleton_weight = None,
    compatibility_gain = None,
    same_tile_support = None,
    incompatibility_penalty = None,
    neighbor_adjacent_tile_support = None,
):
    """Update runtime-configurable DS parameters for experiment sweeps."""
    global DS_TOP_K, DS_MAX_NEIGHBORS, DS_MASS_FUNCTION_MODE, DS_SINGLETON_WEIGHT, DS_NEIGHBOR_COMPATIBILITY_GAIN
    global DS_NEIGHBOR_SAME_TILE_SUPPORT, DS_NEIGHBOR_INCOMPATIBILITY_PENALTY
    global DS_NEIGHBOR_ADJACENT_TILE_SUPPORT

    if top_k is not None:
        DS_TOP_K = max(1, int(top_k))
    if max_neighbors is not None:
        DS_MAX_NEIGHBORS = max(0, int(max_neighbors))
    if mass_function_mode is not None:
        if mass_function_mode not in ("spatial_topk", "topk"):
            raise ValueError(f"Unsupported mass function mode: {mass_function_mode}")
        DS_MASS_FUNCTION_MODE = str(mass_function_mode)
    if singleton_weight is not None:
        DS_SINGLETON_WEIGHT = float(np.clip(singleton_weight, 0.0, 1.0))
    if compatibility_gain is not None:
        DS_NEIGHBOR_COMPATIBILITY_GAIN = float(max(0.0, compatibility_gain))
    if same_tile_support is not None:
        DS_NEIGHBOR_SAME_TILE_SUPPORT = float(np.clip(same_tile_support, 0.0, 1.0))
    if incompatibility_penalty is not None:
        DS_NEIGHBOR_INCOMPATIBILITY_PENALTY = float(np.clip(incompatibility_penalty, 0.0, 1.0))
    if neighbor_adjacent_tile_support is not None:
        DS_NEIGHBOR_ADJACENT_TILE_SUPPORT = float(np.clip(neighbor_adjacent_tile_support, 0.0, 1.0))

def dempster_shafer(
    personal_information,
    previous_mass,
    social_masses,
    theta_set,
    agent_id = None,
    social_fusion_rule = "comp_boost",
    personal_fusion_rule = "dempster",
    social_neighbor_ids = None,
    swarm_grid_matrix = None,
):
    """Fuse personal and social evidence using selectable personal fusion and social fusion.

    Args:
        personal_information (numpy.ndarray): Latest personal observation scores.
        previous_mass (dict): Agent mass from the previous decision loop.
        social_masses (list[dict]): Neighbor masses collected in the receiving phase.
        theta_set (iterable): Frame of discernment.

    Returns:
        tuple[dict, numpy.ndarray]: Pruned fused mass and per-tile decision scores.
    """
    number_of_choices = len(personal_information)
    theta_set = _ensure_theta(theta_set, number_of_choices)
    grid_shape = _infer_grid_shape(number_of_choices)

    if DS_MASS_FUNCTION_MODE == "spatial_topk":
        personal_mass = _build_spatial_topk_personal_mass(personal_information, theta_set, grid_shape)
    elif DS_MASS_FUNCTION_MODE == "topk":
        personal_mass = _build_topk_personal_mass(personal_information, theta_set)
    else:
        raise ValueError(f"Unsupported mass function mode: {DS_MASS_FUNCTION_MODE}")

    selected_personal_fusion_rule = personal_fusion_rule
    if personal_fusion_rule == "yager":
        fused_mass = _yager_rule_fusion(previous_mass, personal_mass, theta_set)
    elif personal_fusion_rule == "dempster":
        fused_mass = _dempster_rule_fusion(previous_mass, personal_mass, theta_set)
    elif personal_fusion_rule == "adaptive":
        conflict_mass = _calculate_conflict_mass(previous_mass, personal_mass)
        if conflict_mass > DS_CONFLICT_MASS_SELECTION_THRESHOLD:
            selected_personal_fusion_rule = "yager"
            fused_mass = _yager_rule_fusion(previous_mass, personal_mass, theta_set)
        else:
            selected_personal_fusion_rule = "dempster"
            fused_mass = _dempster_rule_fusion(previous_mass, personal_mass, theta_set)
    else:
        raise ValueError(f"Unsupported personal fusion rule: {personal_fusion_rule}")
    if social_fusion_rule == "yager":
        for neighbor_mass in social_masses:
            fused_mass = _yager_rule_fusion(fused_mass, neighbor_mass, theta_set)
    elif social_fusion_rule in ("comp_boost", "comp_boost_directed"):
        use_directed_adjacency = social_fusion_rule == "comp_boost_directed"
        fused_mass = _apply_neighbor_compatibility_boost(
            fused_mass,
            social_masses,
            theta_set,
            grid_shape,
            social_neighbor_ids = social_neighbor_ids,
            agent_id = agent_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
    else:
        raise ValueError(f"Unsupported social fusion rule: {social_fusion_rule}")

    fused_mass = _prune_mass(fused_mass, theta_set)
    decision_scores = _compute_decision_scores(fused_mass, number_of_choices)
    best_element_label, best_element_confidence = _get_best_element_for_debug(
        fused_mass,
        grid_shape,
        number_of_choices,
    )
    # print(f"[agent {agent_id}] best element: {best_element_label}, confidence: {best_element_confidence:.6f}")
    return fused_mass, decision_scores, selected_personal_fusion_rule

def _get_best_element_for_debug(mass_dict, grid_shape, number_of_choices):
    """Return the strongest non-Theta focal set and its confidence."""
    rows, _ = grid_shape
    best_label = "None"
    best_confidence = -1.0

    for focal_set, mass in mass_dict.items():
        if len(focal_set) == 0 or len(focal_set) == number_of_choices:
            continue

        sorted_indices = sorted(focal_set)
        sorted_names = [_index_to_subarea_name(index, rows) for index in sorted_indices]
        focal_label = sorted_names[0] if len(sorted_names) == 1 else "{" + ", ".join(sorted_names) + "}"
        focal_confidence = float(mass)
        if focal_confidence > best_confidence:
            best_label = focal_label
            best_confidence = focal_confidence

    return best_label, best_confidence

def _index_to_subarea_name(index, rows):
    """Convert a codebook index into the corresponding subarea name."""
    x_index = int(index) // rows
    y_index = int(index) % rows
    return f"{x_index + 1:02d}_{y_index + 1:02d}"

def _ensure_theta(theta_set, number_of_choices):
    """Return Theta as an immutable set spanning the full hypothesis space."""
    if len(theta_set) != number_of_choices:
        return frozenset(range(number_of_choices))
    return frozenset(theta_set)

def _build_spatial_topk_personal_mass(personal_information, theta_set, grid_shape):
    """Build top-k singleton and neighbor-selected pair focal sets from the latest personal observation."""
    k = min(DS_TOP_K, len(personal_information))
    if k <= 0:
        return {theta_set: 1.0}

    top_k_indices = np.argsort(personal_information)[-k:][::-1]
    normalized_belief = _normalize_within_top_k(personal_information[top_k_indices])

    confidence = _get_entropy_based_confidence(personal_information)
    singleton_mass_scale = confidence * DS_SINGLETON_WEIGHT
    pair_mass_scale = confidence * (1 - DS_SINGLETON_WEIGHT)
    theta_mass = 1 - confidence

    personal_mass = {theta_set: float(theta_mass)}
    for rank, hypothesis_index in enumerate(top_k_indices):
        focal_singleton = frozenset({int(hypothesis_index)})
        belief_weight = normalized_belief[rank]
        personal_mass[focal_singleton] = personal_mass.get(focal_singleton, 0.0) + float(singleton_mass_scale * belief_weight)

        pair_sets = _build_neighbor_pairs(
            int(hypothesis_index),
            personal_information,
            grid_shape,
            len(personal_information),
        )
        if not pair_sets:
            continue

        pair_mass_share = float(pair_mass_scale * belief_weight) / len(pair_sets)
        for pair_set in pair_sets:
            personal_mass[pair_set] = personal_mass.get(pair_set, 0.0) + pair_mass_share

    return _normalize_mass(personal_mass)

def _build_topk_personal_mass(personal_information, theta_set):
    """Build a personal mass from top-k singletons, all their pairs, one joint top-k set, and Theta."""
    k = min(DS_TOP_K, len(personal_information))
    if k <= 0:
        return {theta_set: 1.0}

    top_k_indices = np.argsort(personal_information)[-k:][::-1]
    normalized_belief = _normalize_within_top_k(personal_information[top_k_indices])

    confidence = _get_entropy_based_confidence(personal_information)
    singleton_mass_scale = confidence * DS_SINGLETON_WEIGHT
    composite_mass_scale = confidence * (1 - DS_SINGLETON_WEIGHT)
    theta_mass = 1 - confidence

    personal_mass = {theta_set: float(theta_mass)}
    singleton_reference_weights = {}
    for rank, hypothesis_index in enumerate(top_k_indices):
        focal_singleton = frozenset({int(hypothesis_index)})
        singleton_mass = float(singleton_mass_scale * normalized_belief[rank])
        singleton_reference_weights[int(hypothesis_index)] = float(normalized_belief[rank])
        personal_mass[focal_singleton] = personal_mass.get(focal_singleton, 0.0) + singleton_mass

    composite_weight_map = {}
    top_k_indices_list = [int(index) for index in top_k_indices]
    for left_index in range(len(top_k_indices_list)):
        for right_index in range(left_index + 1, len(top_k_indices_list)):
            hypothesis_a = top_k_indices_list[left_index]
            hypothesis_b = top_k_indices_list[right_index]
            pair_set = frozenset({hypothesis_a, hypothesis_b})
            composite_weight_map[pair_set] = (
                singleton_reference_weights[hypothesis_a] + singleton_reference_weights[hypothesis_b]
            )

    if len(top_k_indices_list) > 1:
        top_k_set = frozenset(top_k_indices_list)
        composite_weight_map[top_k_set] = sum(
            singleton_reference_weights[hypothesis_index]
            for hypothesis_index in top_k_indices_list
        )

    total_composite_weight = sum(composite_weight_map.values())
    if total_composite_weight > 0.0 and composite_mass_scale > 0.0:
        for focal_set, composite_weight in composite_weight_map.items():
            personal_mass[focal_set] = personal_mass.get(focal_set, 0.0) + float(
                composite_mass_scale * (composite_weight / total_composite_weight)
            )

    return _normalize_mass(personal_mass)

def _prune_mass(mass_dict, theta_set):
    """Keep the strongest non-Theta focal sets in one global pruning pass."""
    theta_set = frozenset(theta_set)
    if not mass_dict:
        return {theta_set: 1.0}

    retention_budget = _get_mass_retention_budget(len(theta_set))
    ranked_focal_sets = sorted(
        (
            (focal_set, mass)
            for focal_set, mass in mass_dict.items()
            if 0 < len(focal_set) < len(theta_set)
        ),
        key = lambda item: item[1],
        reverse = True,
    )

    kept_focal_sets = set()
    pruned_mass = {}
    for focal_set, mass in ranked_focal_sets[:retention_budget]:
        kept_focal_sets.add(focal_set)
        pruned_mass[focal_set] = mass

    theta_mass = mass_dict.get(theta_set, 0.0)
    discarded_mass = sum(
        mass
        for focal_set, mass in mass_dict.items()
        if focal_set not in kept_focal_sets and focal_set != theta_set
    )
    pruned_mass[theta_set] = theta_mass + discarded_mass

    return _normalize_mass(pruned_mass)

def _get_mass_retention_budget(number_of_choices):
    """Return the maximum number of non-Theta focal sets expected for the active mass construction mode."""
    top_k = min(max(1, int(DS_TOP_K)), int(number_of_choices))
    if DS_MASS_FUNCTION_MODE == "spatial_topk":
        return top_k + top_k * max(0, int(DS_MAX_NEIGHBORS))
    if DS_MASS_FUNCTION_MODE == "topk":
        pair_count = comb(top_k, 2) if top_k >= 2 else 0
        top_k_set_count = 1 if top_k >= 2 else 0
        return top_k + pair_count + top_k_set_count
    raise ValueError(f"Unsupported mass function mode: {DS_MASS_FUNCTION_MODE}")

def _compute_decision_scores(mass_dict, number_of_choices):
    """Project the fused mass function to tile-level decision scores."""
    decision_scores = np.zeros(number_of_choices, dtype = np.float32)
    for focal_set, mass in mass_dict.items():
        if not focal_set or len(focal_set) == number_of_choices:
            continue
        for hypothesis_index in focal_set:
            decision_scores[hypothesis_index] += mass
    return decision_scores

def _apply_neighbor_compatibility_boost(
    mass_dict,
    social_masses,
    theta_set,
    grid_shape,
    social_neighbor_ids = None,
    agent_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Apply the original best-singleton compatibility boost and penalty."""
    if not social_masses:
        return mass_dict

    theta_set = frozenset(theta_set)
    theta_size = len(theta_set)
    if social_neighbor_ids is None:
        social_neighbor_ids = [None] * len(social_masses)

    neighbor_entries = [
        (neighbor_mass, social_neighbor_ids[index] if index < len(social_neighbor_ids) else None)
        for index, neighbor_mass in enumerate(social_masses)
        if any(0 < len(focal_set) < theta_size and mass > 0.0 for focal_set, mass in neighbor_mass.items())
    ]
    if not neighbor_entries:
        return mass_dict

    neighbor_best_entries = []
    for neighbor_mass, neighbor_id in neighbor_entries:
        neighbor_best_singleton = _get_best_singleton_from_mass(neighbor_mass)
        if neighbor_best_singleton is None:
            continue
        neighbor_best_entries.append((int(neighbor_best_singleton), neighbor_id))
    if not neighbor_best_entries:
        return mass_dict

    updated_mass = {}
    for focal_set, mass in mass_dict.items():
        if len(focal_set) == 0 or len(focal_set) == theta_size:
            updated_mass[focal_set] = mass
            continue

        compatibility_support = _compute_neighbor_best_singleton_support(
            focal_set,
            neighbor_best_entries,
            grid_shape,
            agent_id = agent_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
        updated_mass[focal_set] = mass * (1.0 + DS_NEIGHBOR_COMPATIBILITY_GAIN * compatibility_support)

    updated_mass = _normalize_mass(updated_mass)

    local_best_singleton = _get_best_singleton_from_mass(updated_mass)
    if local_best_singleton is None:
        return updated_mass

    local_best_focal_set = frozenset({int(local_best_singleton)})
    is_supported = any(
        _compute_neighbor_best_singleton_compatibility(
            local_best_focal_set,
            neighbor_best_singleton,
            grid_shape,
            agent_id = agent_id,
            neighbor_id = neighbor_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        ) > 0.0
        for neighbor_best_singleton, neighbor_id in neighbor_best_entries
    )
    if is_supported:
        return updated_mass

    if local_best_focal_set in updated_mass:
        updated_mass[local_best_focal_set] *= DS_NEIGHBOR_INCOMPATIBILITY_PENALTY
    return _normalize_mass(updated_mass)

def _get_best_singleton_from_mass(mass_dict):
    """Return the strongest singleton from a DS mass function."""
    best_singleton = None
    best_mass = -1.0
    for focal_set, mass in mass_dict.items():
        if len(focal_set) != 1:
            continue
        if mass > best_mass:
            best_singleton = next(iter(focal_set))
            best_mass = float(mass)
    return best_singleton

def _compute_neighbor_best_singleton_compatibility(
    local_focal_set,
    neighbor_best_singleton,
    grid_shape,
    agent_id = None,
    neighbor_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Return compatibility between one local focal set and one neighbor best singleton."""
    neighbor_focal_set = frozenset({int(neighbor_best_singleton)})
    return _compute_focal_set_compatibility(
        local_focal_set,
        neighbor_focal_set,
        grid_shape,
        agent_id = agent_id,
        neighbor_id = neighbor_id,
        swarm_grid_matrix = swarm_grid_matrix,
        use_directed_adjacency = use_directed_adjacency,
    )

def _compute_neighbor_best_singleton_support(
    local_focal_set,
    neighbor_best_entries,
    grid_shape,
    agent_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Average compatibility of one local focal set against neighbors' best singletons."""
    if not neighbor_best_entries:
        return 0.0

    support_values = [
        _compute_neighbor_best_singleton_compatibility(
            local_focal_set,
            neighbor_best_singleton,
            grid_shape,
            agent_id = agent_id,
            neighbor_id = neighbor_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
        for neighbor_best_singleton, neighbor_id in neighbor_best_entries
    ]
    return float(np.mean(support_values, dtype = np.float32))

def _compute_focal_set_compatibility(
    focal_set_a,
    focal_set_b,
    grid_shape,
    agent_id = None,
    neighbor_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Return overlap/adjacency compatibility between two focal sets."""
    if focal_set_a & focal_set_b:
        return float(DS_NEIGHBOR_SAME_TILE_SUPPORT)

    for hypothesis_index_a in focal_set_a:
        for hypothesis_index_b in focal_set_b:
            if not _are_adjacent_tiles(hypothesis_index_a, hypothesis_index_b, grid_shape):
                continue
            if use_directed_adjacency and not _is_directionally_compatible_adjacent_pair(
                hypothesis_index_a,
                hypothesis_index_b,
                grid_shape,
                agent_id,
                neighbor_id,
                swarm_grid_matrix,
            ):
                continue
            return float(DS_NEIGHBOR_ADJACENT_TILE_SUPPORT)
    return 0.0

def _compute_focal_set_neighbor_support(
    focal_set,
    neighbor_mass,
    theta_set,
    grid_shape,
    agent_id = None,
    neighbor_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Return mass-weighted support of one neighbor for one local focal set."""
    theta_size = len(theta_set)
    support = 0.0
    for neighbor_focal_set, neighbor_focal_mass in neighbor_mass.items():
        if len(neighbor_focal_set) == 0 or len(neighbor_focal_set) == theta_size:
            continue
        support += neighbor_focal_mass * _compute_focal_set_compatibility(
            focal_set,
            neighbor_focal_set,
            grid_shape,
            agent_id = agent_id,
            neighbor_id = neighbor_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
    return float(support)

def _compute_focal_set_social_support(
    focal_set,
    neighbor_entries,
    theta_set,
    grid_shape,
    agent_id = None,
    swarm_grid_matrix = None,
    use_directed_adjacency = False,
):
    """Average focal-set support over all neighbors."""
    if not neighbor_entries:
        return 0.0

    support_values = [
        _compute_focal_set_neighbor_support(
            focal_set,
            neighbor_mass,
            theta_set,
            grid_shape,
            agent_id = agent_id,
            neighbor_id = neighbor_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
        for neighbor_mass, neighbor_id in neighbor_entries
    ]
    return float(np.mean(support_values, dtype = np.float32))

def _infer_grid_shape(number_of_choices):
    """Infer the x-major grid layout of the terrain tile codebook."""
    factor_pairs = [
        (rows, number_of_choices // rows)
        for rows in range(1, int(np.sqrt(number_of_choices)) + 1)
        if number_of_choices % rows == 0
    ]
    if not factor_pairs:
        return 1, number_of_choices
    return min(factor_pairs, key = lambda pair: (abs(pair[1] - pair[0]), pair[0]))

def _build_neighbor_pairs(index, personal_information, grid_shape, number_of_choices):
    """Build pair focal sets with the strongest adjacent neighbors of one hypothesis."""
    rows, cols = grid_shape
    row = index % rows
    col = index // rows

    candidate_neighbors = []
    for row_offset, col_offset in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ):
        neighbor_row = row + row_offset
        neighbor_col = col + col_offset
        if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
            neighbor_index = neighbor_col * rows + neighbor_row
            if neighbor_index < number_of_choices:
                candidate_neighbors.append(int(neighbor_index))

    if not candidate_neighbors:
        return []

    max_neighbors = min(max(0, int(DS_MAX_NEIGHBORS)), len(candidate_neighbors))
    if max_neighbors == 0:
        return []

    ranked_neighbors = sorted(
        candidate_neighbors,
        key = lambda neighbor_index: (float(personal_information[neighbor_index]), -neighbor_index),
        reverse = True,
    )
    selected_neighbors = ranked_neighbors[:max_neighbors]
    return [frozenset({index, neighbor_index}) for neighbor_index in selected_neighbors]

def _are_adjacent_tiles(index_a, index_b, grid_shape):
    """Return True when two tiles are 8-connected neighbors."""
    rows, _ = grid_shape
    row_a = index_a % rows
    col_a = index_a // rows
    row_b = index_b % rows
    col_b = index_b // rows
    return max(abs(row_a - row_b), abs(col_a - col_b)) == 1

def _get_swarm_grid_position(agent_id, swarm_grid_matrix):
    """Return row/col of one agent id inside the shared swarm grid."""
    if swarm_grid_matrix is None or agent_id is None:
        return None

    matches = np.argwhere(swarm_grid_matrix == int(agent_id))
    if len(matches) == 0:
        return None
    row_index, col_index = matches[0]
    return int(row_index), int(col_index)

def _get_tile_grid_position(hypothesis_index, grid_shape):
    """Return row/col of one hypothesis tile in the terrain grid."""
    rows, _ = grid_shape
    return int(hypothesis_index) % rows, int(hypothesis_index) // rows

def _sign_tuple(row_delta, col_delta):
    """Reduce one 2D offset to its directional sign components."""
    return int(np.sign(row_delta)), int(np.sign(col_delta))

def _is_directionally_compatible_adjacent_pair(
    local_hypothesis_index,
    neighbor_hypothesis_index,
    grid_shape,
    agent_id,
    neighbor_id,
    swarm_grid_matrix,
):
    """Return whether one adjacent tile pair matches the neighbor's relative swarm direction."""
    local_agent_position = _get_swarm_grid_position(agent_id, swarm_grid_matrix)
    neighbor_agent_position = _get_swarm_grid_position(neighbor_id, swarm_grid_matrix)
    if local_agent_position is None or neighbor_agent_position is None:
        return False

    agent_row_delta = neighbor_agent_position[0] - local_agent_position[0]
    agent_col_delta = neighbor_agent_position[1] - local_agent_position[1]
    expected_direction = _sign_tuple(agent_row_delta, agent_col_delta)
    if expected_direction == (0, 0):
        return False

    local_tile_position = _get_tile_grid_position(local_hypothesis_index, grid_shape)
    neighbor_tile_position = _get_tile_grid_position(neighbor_hypothesis_index, grid_shape)
    tile_row_delta = neighbor_tile_position[0] - local_tile_position[0]
    tile_col_delta = neighbor_tile_position[1] - local_tile_position[1]
    actual_direction = _sign_tuple(tile_row_delta, tile_col_delta)

    return actual_direction == expected_direction

def _normalize_mass(mass_dict):
    """Normalize a mass function so all focal-set masses sum to one."""
    total_mass = sum(mass_dict.values())
    if total_mass <= 0:
        return mass_dict
    return {focal_set: mass / total_mass for focal_set, mass in mass_dict.items()}

def _calculate_conflict_mass(mass_a, mass_b):
    """Return the DS conflict mass between two mass functions."""
    if not mass_a or not mass_b:
        return 0.0

    conflict_mass = 0.0
    for hypothesis_a, mass_value_a in mass_a.items():
        focal_set_a = frozenset(hypothesis_a)
        for hypothesis_b, mass_value_b in mass_b.items():
            focal_set_b = frozenset(hypothesis_b)
            if not (focal_set_a & focal_set_b):
                conflict_mass += mass_value_a * mass_value_b
    return float(conflict_mass)

def _yager_rule_fusion(mass_a, mass_b, theta_set):
    """Combine two mass functions using Yager's rule."""
    if not mass_a:
        return mass_b

    theta_set = frozenset(theta_set)
    if not mass_b:
        raise ValueError("mass_b must not be empty.")

    combined_mass = {}
    conflict_mass = 0.0
    eps = 1e-12

    for hypothesis_a, mass_value_a in mass_a.items():
        focal_set_a = frozenset(hypothesis_a)
        for hypothesis_b, mass_value_b in mass_b.items():
            focal_set_b = frozenset(hypothesis_b)
            product_mass = mass_value_a * mass_value_b
            intersection = focal_set_a & focal_set_b
            if not intersection:
                conflict_mass += product_mass
            else:
                combined_mass[intersection] = combined_mass.get(intersection, 0.0) + product_mass

    combined_mass[theta_set] = combined_mass.get(theta_set, 0.0) + conflict_mass
    combined_mass = {focal_set: mass for focal_set, mass in combined_mass.items() if mass > eps}

    total_mass = sum(combined_mass.values())
    if total_mass > eps:
        combined_mass = {focal_set: mass / total_mass for focal_set, mass in combined_mass.items()}
    return combined_mass

def _dempster_rule_fusion(mass_a, mass_b, theta_set):
    """Combine two mass functions using Dempster's normalized conjunctive rule."""
    if not mass_a:
        return mass_b

    theta_set = frozenset(theta_set)
    if not mass_b:
        raise ValueError("mass_b must not be empty.")

    combined_mass = {}
    eps = 1e-12

    for hypothesis_a, mass_value_a in mass_a.items():
        focal_set_a = frozenset(hypothesis_a)
        for hypothesis_b, mass_value_b in mass_b.items():
            focal_set_b = frozenset(hypothesis_b)
            intersection = focal_set_a & focal_set_b
            if not intersection:
                continue
            product_mass = mass_value_a * mass_value_b
            combined_mass[intersection] = combined_mass.get(intersection, 0.0) + product_mass

    combined_mass = {focal_set: mass for focal_set, mass in combined_mass.items() if mass > eps}
    total_mass = sum(combined_mass.values())
    if total_mass <= eps:
        return {theta_set: 1.0}

    return {focal_set: mass / total_mass for focal_set, mass in combined_mass.items()}

def _normalize_within_top_k(top_k_belief):
    """Normalize the selected top-k belief values."""
    total_belief = np.sum(top_k_belief)
    if total_belief == 0:
        return np.zeros_like(top_k_belief)
    return top_k_belief / total_belief

def _get_entropy_based_confidence(belief):
    """Convert entropy to a confidence value in [0, 1]."""
    epsilon = 1e-12
    belief = np.clip(belief, epsilon, 1 - epsilon)
    entropy = -np.sum(belief * np.log(belief))
    return 1 - (entropy / np.log(len(belief)))
