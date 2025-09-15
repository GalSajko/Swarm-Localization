import numpy as np

import constants as const
import utils

def calculate_personal_information(cosine_similarities):
    softmax_probabilities = utils.softmax_over_vector(cosine_similarities)
    personal_info = softmax_probabilities / (1 - softmax_probabilities + 1e-12)

    return np.clip(personal_info, 0, 6.5)

def calculate_social_information(deltas, social_factor):
    social_info = np.e ** (social_factor * deltas) - 1
    return np.clip(social_info, -3.0, 6.5)

def calculate_asodm_parameters(all_opinions, no_of_choices):
    if not all_opinions.size:
        deltas = np.zeros(no_of_choices)
        betas = np.ones(no_of_choices) * 5
        weights = np.zeros(no_of_choices)

        return deltas, betas, weights
    
    if all_opinions.shape[0] > 1:
        weights = np.mean(all_opinions, axis = 0) / const.DECISION_THRESHOLD
    else:
        weights = all_opinions[0] / const.DECISION_THRESHOLD

    weights[np.where(weights > 1.0)] = 1.0
    weights[np.where(weights < -1.0)] = -1.0

    max_counts = np.zeros(no_of_choices)
    decision_counts = np.zeros(no_of_choices)
    for opinions in all_opinions:
        max_index = np.argmax(opinions)
        max_counts[max_index] += 1
        decision_counts[np.where(opinions >= const.DECISION_THRESHOLD)] += 1

    deltas = np.zeros(no_of_choices)
    for i, weight in enumerate(weights):
        other_weights = np.delete(weights, i)
        other_max_counts = np.delete(max_counts, i)
        deltas[i] = weight * max_counts[i] - np.dot(other_weights, other_max_counts)
    
    betas = 5 * np.exp(-1 * decision_counts)

    return deltas, betas, weights

def asodm(personal_information, social_factor, all_opinions, no_of_choices):
    deltas, betas, weights = calculate_asodm_parameters(all_opinions, no_of_choices)
    social_information = calculate_social_information(deltas, social_factor)
    # social_information = np.zeros_like(personal_information)
    # social_information = weights
    # print("====")
    # print(deltas)
    # print(personal_information)
    # print(social_information)
    # print("======")
    updated_opinion = betas * personal_information + social_information

    return updated_opinion