import numpy as np

import constants as const
import utils

def calculate_personal_information(cosine_similarities):
    softmax_probabilities = utils.softmax_over_vector(cosine_similarities)
    personal_info = softmax_probabilities / (1 - softmax_probabilities + 1e-12)

    return np.clip(personal_info, 0, 6.5)

def calculate_social_information(deltas, social_factor):
    social_info = np.e ** (social_factor * deltas) - 1
    return np.clip(social_info, -6.5, 6.5)

def calculate_asodm_parameters(all_opinions, no_of_choices, decision_threshold):
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
    deltas, betas = calculate_asodm_parameters(all_opinions, no_of_choices, decision_threshold)
    social_information = calculate_social_information(deltas, social_factor)
    
    updated_opinion = betas * personal_information + social_information

    return updated_opinion