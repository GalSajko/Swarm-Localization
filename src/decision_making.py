import numpy as np

import constants as const
import utils

def calculate_personal_information(cosine_similarities):
    softmax_probabilities = utils.softmax_over_vector(cosine_similarities)
    personal_info = softmax_probabilities / (1 - softmax_probabilities + 1e-12)
    # print(np.argmax(personal_info), np.max(personal_info))
    return np.clip(personal_info, 0, 6.5)

def calculate_social_information(deltas, social_factor):
    social_info = np.e ** (social_factor * deltas) - 1
    return np.clip(social_info, -6.5, 6.5)

def calculate_asodm_parameters(all_opinions, no_of_choices):
    if not all_opinions.size:
        deltas = np.zeros(no_of_choices)
        betas = np.ones(no_of_choices) * 5
        weights = np.zeros(no_of_choices)

        return deltas, betas, weights

    weights = np.mean(all_opinions, axis = 0) / const.DECISION_THRESHOLD
    weights = np.clip(weights, -1.0, 1.0)

    max_counts = np.zeros(no_of_choices)
    decision_counts = np.zeros(no_of_choices)
    for opinions in all_opinions:
        max_counts[np.argmax(opinions)] += 1
        decision_counts[opinions >= const.DECISION_THRESHOLD] += 1

    wm = weights * max_counts
    deltas = wm - (np.sum(wm) - wm) / (len(wm) - 1)

    # betas = 100 * np.exp(decision_counts / const.NUMBER_OF_AGENTS)
    betas = 185

    return deltas, betas, weights

def asodm(personal_information, social_factor, all_opinions, no_of_choices):
    deltas, betas, weights = calculate_asodm_parameters(all_opinions, no_of_choices)
    social_information = calculate_social_information(deltas, social_factor)

    np.seterr(all = 'raise')
    ratio = 0.0
    try:
        ratio = np.abs(personal_information / social_information)[-1]
    except:
        pass

    # print(np.argmax(betas), np.argmax(social_information))
    
    updated_opinion = betas * personal_information + social_information

    return updated_opinion, personal_information[-1], social_information[-1], ratio