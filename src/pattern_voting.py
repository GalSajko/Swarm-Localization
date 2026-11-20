"""Swarm pattern vote with fuzzy memberships.

Every agent contributes the memberships of its selected feature, which lies at the agent's position after re-centering, and its position relative
to a reference agent, measured directly between the agents. Each candidate map feature of an agent votes for the position the reference agent would
have if that candidate were the agent's true feature. Places where votes of several agents agree within MATCHING_RADIUS are scored by the sum over
agents of each agent's best overlap there, and the best place is the swarm's estimate of the reference agent's position.
"""

import numpy as np
from scipy.spatial import cKDTree

import constants

def select_best_vote_per_agent(vote_indices, vote_agents, vote_weights):
    """Keep, for each agent, its highest-weighted vote among the given votes.

    Returns:
        numpy.ndarray: Indices of the kept votes, one per agent.
    """
    vote_indices = np.asarray(vote_indices, dtype = np.int64)
    if len(vote_indices) == 0:
        return vote_indices
    order = vote_indices[np.lexsort((-vote_weights[vote_indices], vote_agents[vote_indices]))]
    is_first_of_agent = np.concatenate(([True], vote_agents[order][1:] != vote_agents[order][:-1]))
    return order[is_first_of_agent]

def vote(relative_positions, memberships, terrain_map, matching_radius = None):
    """Find the position of the reference agent on which most agents agree.

    Args:
        relative_positions (numpy.ndarray): Positions of the agents relative to the reference agent (shape (N, 2), map pixels, row 0 is usually the
            reference itself with (0, 0)).
        memberships (numpy.ndarray): Memberships of the agents' selected features, shape (N, C).
        terrain_map (map_matching.TerrainMap): Terrain codebook.
        matching_radius (float, optional): Radius in map pixels within which votes agree. Defaults to constants.MATCHING_RADIUS.

    Returns:
        dict: 'reference_position' (estimated map position of the reference agent, or None if no agent has a candidate), 'score', and
            'agreeing_agents' (indices of the agents that agree at the estimate).
    """
    matching_radius = constants.MATCHING_RADIUS if matching_radius is None else matching_radius
    vote_positions, vote_agents, vote_weights = [], [], []
    for agent_index, (relative_position, agent_memberships) in enumerate(zip(relative_positions, memberships)):
        candidate_indices, overlaps = terrain_map.find_candidates(agent_memberships)
        if len(candidate_indices):
            vote_positions.append(terrain_map.keypoints_xy[candidate_indices] - relative_position)
            vote_agents.append(np.full(len(candidate_indices), agent_index))
            vote_weights.append(overlaps)

    if not vote_positions:
        return {"reference_position": None, "score": 0.0, "agreeing_agents": np.zeros(0, dtype = np.int64)}

    vote_positions = np.vstack(vote_positions)
    vote_agents = np.concatenate(vote_agents)
    vote_weights = np.concatenate(vote_weights)
    tree = cKDTree(vote_positions)

    # Without any agreement between agents, the single strongest vote is the answer.
    best_votes = np.array([int(np.argmax(vote_weights))])
    best_score = float(vote_weights[best_votes[0]])

    # Only votes that agree with a vote of another agent can form a place supported by more than one agent.
    pairs = tree.query_pairs(matching_radius, output_type = "ndarray")
    if len(pairs):
        pairs = pairs[vote_agents[pairs[:, 0]] != vote_agents[pairs[:, 1]]]
    if len(pairs):
        candidate_places = np.unique(pairs.ravel())
        for neighbors in tree.query_ball_point(vote_positions[candidate_places], matching_radius):
            kept_votes = select_best_vote_per_agent(neighbors, vote_agents, vote_weights)
            score = float(vote_weights[kept_votes].sum())
            if score > best_score:
                best_score, best_votes = score, kept_votes

    weights = vote_weights[best_votes]
    reference_position = (vote_positions[best_votes] * weights[:, None]).sum(axis = 0) / weights.sum()
    return {"reference_position": reference_position, "score": best_score, "agreeing_agents": np.sort(vote_agents[best_votes])}
