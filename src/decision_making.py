import numpy as np

DS_TOP_K = 4
DS_SINGLETON_WEIGHT = 0.5
DS_MAX_PAIR_FOCAL_SETS = 8
DS_NEIGHBOR_COMPATIBILITY_GAIN = 1.0
DS_NEIGHBOR_SAME_TILE_SUPPORT = 1.0
DS_NEIGHBOR_ADJACENT_TILE_SUPPORT = 0.8
DS_NEIGHBOR_INCOMPATIBILITY_PENALTY = 0.1

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

def dempster_shafer(personal_information, previous_mass, social_masses, theta_set, agent_id = None, use_yager_social_fusion = False):
    """Fuse personal and social evidence using Compatibility boost or Yager's rule.

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

    fused_mass = _yager_rule_fusion(previous_mass, personal_mass, theta_set)
    fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)

    if use_yager_social_fusion:
        for neighbor_mass in social_masses:
            pruned_neighbor_mass = _prune_mass(neighbor_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)
            fused_mass = _yager_rule_fusion(fused_mass, pruned_neighbor_mass, theta_set)
            fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)
    else:
        # Alternative neighbor handling for comparison:
        # instead of fully fusing neighbor evidence, softly boost personal focal sets
        # when they spatially align with neighbors' strongest singleton hypotheses.
        fused_mass = _apply_neighbor_compatibility_boost(fused_mass, social_masses, theta_set, grid_shape)
        fused_mass = _prune_mass(fused_mass, DS_TOP_K, DS_MAX_PAIR_FOCAL_SETS, theta_set)

    decision_scores = _compute_decision_scores(fused_mass, number_of_choices)
    best_element_label, best_element_confidence = _get_best_element_for_debug(
        fused_mass,
        grid_shape,
        number_of_choices,
    )
    # print(f"[agent {agent_id}] best element: {best_element_label}, confidence: {best_element_confidence:.6f}")
    return fused_mass, decision_scores

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

def _apply_neighbor_compatibility_boost(mass_dict, social_masses, theta_set, grid_shape):
    """Softly reweight focal sets based on neighbors' strongest singleton hypotheses."""
    if not social_masses:
        return mass_dict

    theta_set = frozenset(theta_set)
    theta_size = len(theta_set)
    neighbor_best_singletons = []
    for neighbor_mass in social_masses:
        best_singleton = _get_best_singleton_from_mass(neighbor_mass)
        if best_singleton is not None:
            neighbor_best_singletons.append(best_singleton)

    if not neighbor_best_singletons:
        return mass_dict

    boosted_mass = {}
    for focal_set, mass in mass_dict.items():
        if len(focal_set) == 0 or len(focal_set) == theta_size:
            boosted_mass[focal_set] = mass
            continue

        compatibility = _compute_focal_set_neighbor_support(
            focal_set,
            neighbor_best_singletons,
            grid_shape,
        )
        boosted_mass[focal_set] = mass * (1.0 + DS_NEIGHBOR_COMPATIBILITY_GAIN * compatibility)

    boosted_mass = _apply_best_singleton_incompatibility_penalty(
        boosted_mass,
        neighbor_best_singletons,
        grid_shape,
    )
    return _normalize_mass(boosted_mass)

def _apply_best_singleton_incompatibility_penalty(mass_dict, neighbor_best_singletons, grid_shape):
    """Apply a mild penalty when the local best singleton is incompatible with all neighbor best singletons."""
    if not neighbor_best_singletons:
        return mass_dict

    best_singleton = _get_best_singleton_from_mass(mass_dict)
    if best_singleton is None:
        return mass_dict

    best_singleton_focal_set = frozenset({best_singleton})
    if best_singleton_focal_set not in mass_dict:
        return mass_dict

    has_compatible_neighbor = any(
        neighbor_singleton == best_singleton
        or _are_adjacent_tiles(best_singleton, neighbor_singleton, grid_shape)
        for neighbor_singleton in neighbor_best_singletons
    )
    if has_compatible_neighbor:
        return mass_dict

    penalized_mass = dict(mass_dict)
    penalized_mass[best_singleton_focal_set] = (
        penalized_mass[best_singleton_focal_set] * DS_NEIGHBOR_INCOMPATIBILITY_PENALTY
    )
    return penalized_mass

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

def _compute_focal_set_neighbor_support(focal_set, neighbor_best_singletons, grid_shape):
    """Average same-tile/adjacency support from neighbors for one focal set."""
    if not neighbor_best_singletons:
        return 0.0

    support_values = []
    for neighbor_singleton in neighbor_best_singletons:
        support = 0.0
        for hypothesis_index in focal_set:
            if neighbor_singleton == hypothesis_index:
                support = max(support, DS_NEIGHBOR_SAME_TILE_SUPPORT)
                continue
            if _are_adjacent_tiles(hypothesis_index, neighbor_singleton, grid_shape):
                support = max(support, DS_NEIGHBOR_ADJACENT_TILE_SUPPORT)
        support_values.append(support)

    return float(np.mean(support_values))

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

def _normalize_mass(mass_dict):
    """Normalize a mass function so all focal-set masses sum to one."""
    total_mass = sum(mass_dict.values())
    if total_mass <= 0:
        return mass_dict
    return {focal_set: mass / total_mass for focal_set, mass in mass_dict.items()}

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
