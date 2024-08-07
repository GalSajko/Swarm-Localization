import numpy as np
NUMBER_OF_AGENTS = 20

AGENT_SIZE = np.array([120, 120])
COMMUNICATION_RANGE = 600

AGENT_COLOR_OPINION_DICT = {
    '1_1.png' : (255, 51, 0),
    '1_2.png' : (255, 191, 0),
    '2_1.png' : (128, 255, 0),
    '2_2.png' : (0, 255, 255),
    '3_1.png' : (0, 64, 255),
    '3_2.png' : (128, 0, 255),
    '4_1.png' : (191, 0, 255),
    '4_2.png' : (255, 0, 255),
    '5_1.png' : (255, 128, 0),
    '5_2.png' : (255, 0, 128)
}

EXPLORATION_STATE = 'e'
RECEIVING_STATE = 'r'
DECISION_MAKING_STATE = 'dm'
DISSEMINATION_STATE = 'd'

RECEIVING_TIME = 5.0
MAX_DISSEMINATION_TIME = 10.0