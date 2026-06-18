import numpy as np

DS_TOP_K = 4
DS_SINGLETON_WEIGHT = 0.9
DS_MAX_PAIR_FOCAL_SETS = 8
DS_NEIGHBOR_COMPATIBILITY_GAIN = 0.75
DS_NEIGHBOR_SAME_TILE_SUPPORT = 0.5
DS_NEIGHBOR_ADJACENT_TILE_SUPPORT = 0.8
DS_NEIGHBOR_INCOMPATIBILITY_PENALTY = 0.9

# Applied only when using the "adaptive" personal fusion rule to select between Dempster's and Yager's rules.
DS_CONFLICT_MASS_SELECTION_THRESHOLD = 0.5

def configure_dempster_shafer(
    top_k = None,
    singleton_weight = None,
    compatibility_gain = None,
    incompatibility_penalty = None,
    neighbor_adjacent_tile_support = None,
):
    """Update runtime-configurable DS parameters for experiment sweeps."""
    global DS_TOP_K, DS_SINGLETON_WEIGHT, DS_NEIGHBOR_COMPATIBILITY_GAIN
    global DS_NEIGHBOR_INCOMPATIBILITY_PENALTY, DS_NEIGHBOR_ADJACENT_TILE_SUPPORT

    if top_k is not None:
        DS_TOP_K = max(1, int(top_k))
    if singleton_weight is not None:
        DS_SINGLETON_WEIGHT = float(np.clip(singleton_weight, 0.0, 1.0))
    if compatibility_gain is not None:
        DS_NEIGHBOR_COMPATIBILITY_GAIN = float(max(0.0, compatibility_gain))
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
    personal_fusion_rule = "yager",
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

    personal_mass = _build_personal_mass_with_pairs(personal_information, theta_set, grid_shape)
    personal_mass = _prune_mass(personal_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)

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
    fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)

    if social_fusion_rule == "yager":
        for neighbor_mass in social_masses:
            pruned_neighbor_mass = _prune_mass(neighbor_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)
            fused_mass = _yager_rule_fusion(fused_mass, pruned_neighbor_mass, theta_set)
            fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)
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
        fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)
    else:
        raise ValueError(f"Unsupported social fusion rule: {social_fusion_rule}")

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

def _build_personal_mass_with_pairs(personal_information, theta_set, grid_shape):
    """Build singleton and pair focal sets from the latest personal observation."""
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

        pair_sets = _build_neighbor_pairs(int(hypothesis_index), grid_shape, len(personal_information))
        if not pair_sets:
            continue

        pair_mass_share = float(pair_mass_scale * belief_weight) / len(pair_sets)
        for pair_set in pair_sets:
            personal_mass[pair_set] = personal_mass.get(pair_set, 0.0) + pair_mass_share

    return _normalize_mass(personal_mass)

def _prune_mass(mass_dict, max_singletons, max_pairs, theta_set):
    """Keep the strongest non-Theta focal sets in one global pruning pass."""
    theta_set = frozenset(theta_set)
    if not mass_dict:
        return {theta_set: 1.0}

    retention_budget = max_singletons + max_pairs
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
    """Apply focal-set-level compatibility boost and incompatibility penalty from neighbor masses."""
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

    boosted_mass = {}
    for focal_set, mass in mass_dict.items():
        if len(focal_set) == 0 or len(focal_set) == theta_size:
            boosted_mass[focal_set] = mass
            continue

        compatibility = _compute_focal_set_social_support(
            focal_set,
            neighbor_entries,
            theta_set,
            grid_shape,
            agent_id = agent_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
        boosted_mass[focal_set] = mass * (1.0 + DS_NEIGHBOR_COMPATIBILITY_GAIN * compatibility)

    boosted_mass = _normalize_mass(boosted_mass)
    penalized_mass = {}
    for focal_set, mass in boosted_mass.items():
        if len(focal_set) == 0 or len(focal_set) == theta_size:
            penalized_mass[focal_set] = mass
            continue

        compatibility = _compute_focal_set_social_support(
            focal_set,
            neighbor_entries,
            theta_set,
            grid_shape,
            agent_id = agent_id,
            swarm_grid_matrix = swarm_grid_matrix,
            use_directed_adjacency = use_directed_adjacency,
        )
        penalty_multiplier = 1.0
        if compatibility <= 0.0:
            penalty_multiplier = 1.0 - DS_NEIGHBOR_INCOMPATIBILITY_PENALTY
        penalized_mass[focal_set] = mass * penalty_multiplier

    return _normalize_mass(penalized_mass)

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

def _build_neighbor_pairs(index, grid_shape, number_of_choices):
    """Build 8-connected pair focal sets centered on one hypothesis."""
    rows, cols = grid_shape
    row = index % rows
    col = index // rows

    pair_sets = []
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
                pair_sets.append(frozenset({index, neighbor_index}))
    return pair_sets

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
