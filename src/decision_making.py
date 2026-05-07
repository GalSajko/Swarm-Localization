import numpy as np

import constants as const
import utils

def calculate_personal_information(cosine_similarities):
    """Calculate personal information based on cosine similarity of agent's VLAD vector and a codebook.

    Args:
        cosine_similarities (numpy.ndarray): Cosine similarities.

    Returns:
        numpy.ndarray: Personal information.
    """
    softmax_probabilities = utils.softmax_over_vector(cosine_similarities)
    personal_info = softmax_probabilities / (1 - softmax_probabilities + 1e-12)

    return np.clip(personal_info, 0, 6.5)

def calculate_social_information(deltas, social_factor):
    """Calculate social information.

    Args:
        deltas (numpy.ndarray): Deltas.
        social_factor (numpy.ndarray): Social factor.

    Returns:
        numpy,ndarray: Social information.
    """
    social_info = np.e ** (social_factor * deltas) - 1
    return np.clip(social_info, -6.5, 6.5)

def calculate_asodm_parameters(all_opinions, no_of_choices, decision_threshold):
    """Calculate ASODM parameters (deltas and betas).

    Args:
        all_opinions (numpy.ndarray): Array of all collected opinions.
        no_of_choices (int): Number of different options.
        decision_threshold (numpy.ndarray): Decision threshold vector.

    Returns:
        (numpy.ndarray, numpy.ndarray): Deltas and betas.
    """
    if not all_opinions.size:
        deltas = np.zeros(no_of_choices)
        betas = np.ones(no_of_choices) * 5

        return deltas, betas

    weights = np.mean(all_opinions, axis = 0) / decision_threshold
    weights = np.clip(weights, -1.0, 1.0)

    max_counts = np.zeros(no_of_choices)
    decision_counts = np.zeros(no_of_choices)
    for opinions in all_opinions:
        max_counts[np.argmax(opinions)] += 1
        decision_counts[opinions >= decision_threshold] += 1

    wm = weights * max_counts
    deltas = wm - (np.sum(wm) - wm) / (len(wm) - 1)

    # betas = 100 * np.exp(decision_counts / const.NUMBER_OF_AGENTS)
    betas = 185

    return deltas, betas

def asodm(personal_information, social_factor, all_opinions, no_of_choices, decision_threshold):
    """ASODM algorithm for combining personal and social information and calculating new opinion.

    Args:
        personal_information (numpy.ndarray): Personal information
        social_factor (floa): Social factor.
        all_opinions (numpy.ndarray): All collected opinions.
        no_of_choices (int): Number of different options.
        decision_threshold (numpy.ndarray): Decision thresholds.

    Returns:
        numpy.ndarray: Updated opinion.
    """
    deltas, betas = calculate_asodm_parameters(all_opinions, no_of_choices, decision_threshold)
    social_information = calculate_social_information(deltas, social_factor)
    
    updated_opinion = betas * personal_information + social_information

    return updated_opinion

def dempster_shafer(personal_information, mass_dict, social_masses, theta_set):
    """Dempster-Shafer theory for combining evidence from cosine similarities.

    Args:
        personal_information (numpy.ndarray): Personal information, defined as softmax over cosine similarities.
        mass_dict (dict): Dictionary containing current mass values for each hypothesis.
        social_masses (list): List of mass dictionaries from neighbors.
        theta_set (iterable): Frame of discernment (universal set).
    Returns:
        numpy.ndarray: Combined belief.
    """
    k = 10
    gamma = 0.5

    # Ensure Theta has the same immutable set structure as all other focal sets.
    if len(theta_set) != len(personal_information):
        theta_set = frozenset(range(len(personal_information)))
    else:
        theta_set = frozenset(theta_set)

    top_k_indices = np.argsort(personal_information)[-k:]
    top_k_hypotheses = [frozenset({int(idx)}) for idx in top_k_indices]
    top_k_belief = personal_information[top_k_indices]
    normalized_top_k_belief = _normalize_within_top_k(top_k_belief)

    confidence = _get_entropy_based_confidence(personal_information)
    alpha = confidence * gamma
    beta = (1 - gamma) * confidence

    top_k_singleton_masses = alpha * normalized_top_k_belief
    top_k_set_mass = beta
    theta_mass = 1 - confidence

    # Masses of the current personal observations.
    mass_personal = {top_k_hypotheses[i]: float(top_k_singleton_masses[i]) for i in range(k)}

    # Set of top-k hypotheses mass.
    top_k_set = frozenset(int(idx) for idx in top_k_indices)
    mass_personal[top_k_set] = float(top_k_set_mass)

    # Ignorance (Theta) mass.
    mass_personal[theta_set] = float(theta_mass)

    # Combine personal mass with previous mass into temporary mass.
    mass_temp = _yager_rule_fusion(mass_dict, mass_personal, theta_set)

    # Combine social masses of neighbors into one social mass.
    # If no neighbors are available, use vacuous social evidence m(Theta) = 1.
    mass_social = {theta_set: 1.0}
    for neighbor_mass in social_masses:
        mass_social = _yager_rule_fusion(mass_social, neighbor_mass, theta_set)

    # Combine temporary mass with social mass into new current mass. 
    new_mass = _yager_rule_fusion(mass_temp, mass_social, theta_set)
    
    # Prune the new mass to keep only top-k singletons and redistribute the rest to the set of top-k hypotheses.
    new_mass = _prune_mass_to_topk_singletons(new_mass, k, theta_set)

    return new_mass

def _prune_mass_to_topk_singletons(mass_dict, k, theta_set):
    """
    Extracts singleton focal elements, ranks them by mass in descending order,
    keeps the top-k singletons, and redistributes the remaining mass to either
    the union of top-k singletons or the frame of discernment. Preserves the
    original mass of the frame of discernment and normalizes the result.
    
    Args:
        mass_dict (dict): Dictionary with frozenset keys (focal elements) and
            float values (masses). Represents a belief mass function.
        k (int): Number of top singletons to keep. Must be positive.
        theta_set (iterable): Frame of discernment (universal set).
    
    Returns:
        dict: Pruned mass function with normalized masses (sum to 1.0).
    
    Raises:
        ValueError: If k is not a positive integer.
    """
    theta_set = frozenset(theta_set)

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Step 1: extract singleton masses
    singletons = [(fs, m) for fs, m in mass_dict.items() if len(fs) == 1]
    singletons.sort(key=lambda x: x[1], reverse=True)

    topk = singletons[:k]

    # Step 2: build new mass
    new_mass = {}

    # Keep top-k singletons
    for fs, m in topk:
        new_mass[fs] = m

    # Step 3: build A_i (top-k set)
    if topk:
        topk_set = frozenset().union(*(fs for fs, _ in topk))
    else:
        topk_set = frozenset()

    # Step 4: collect all remaining non-Theta mass
    remaining_mass = 0.0
    for fs, m in mass_dict.items():
        if fs not in new_mass and fs != theta_set:
            remaining_mass += m

    # Assign remaining non-Theta mass to A_i only if it is non-empty.
    if topk_set:
        new_mass[topk_set] = new_mass.get(topk_set, 0.0) + remaining_mass
    else:
        new_mass[theta_set] = new_mass.get(theta_set, 0.0) + remaining_mass

    # Step 5: Theta keeps its original mass.
    theta_mass = mass_dict.get(theta_set, 0.0)
    new_mass[theta_set] = new_mass.get(theta_set, 0.0) + theta_mass

    # Step 6: normalize
    total = sum(new_mass.values())
    if total > 0:
        new_mass = {fs: m / total for fs, m in new_mass.items()}

    return new_mass

def _yager_rule_fusion(mass_a, mass_b, theta_set):
    """Yager's rule of combination for Dempster-Shafer theory.

    Args:
        mass_a (dict): Mass function A.
        mass_b (dict): Mass function B.
    
    Returns:
        dict: Combined mass function.
    """
    if not mass_a:
        return mass_b
    
    theta_set = frozenset(theta_set)

    if not mass_a:
        raise ValueError("mass_a must not be empty.")
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

    combined_mass = {k: v for k, v in combined_mass.items() if v > eps}

    total = sum(combined_mass.values())
    if total > eps:
        combined_mass = {k: v / total for k, v in combined_mass.items()}

    return combined_mass

def _normalize_within_top_k(top_k_belief):
    """Normalize belief values within the top-k indices.

    Args:
        top_k_belief (numpy.ndarray): Belief values of the top-k indices.

    Returns:
        numpy.ndarray: Normalized belief values.
    """
    total_belief = np.sum(top_k_belief)
    if total_belief == 0:
        return np.zeros_like(top_k_belief)
    
    normalized_belief = top_k_belief / total_belief

    return normalized_belief

def _get_entropy_based_confidence(belief):
    """Calculate confidence based on entropy of belief distribution.

    Args:
        belief (numpy.ndarray): Belief distribution.

    Returns:
        float: Confidence value.
    """
    epsilon = 1e-12
    belief = np.clip(belief, epsilon, 1 - epsilon)
    entropy = -np.sum(belief * np.log(belief))
    confidence = 1 - (entropy / np.log(len(belief)))
    
    return confidence
