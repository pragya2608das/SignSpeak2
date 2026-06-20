import numpy as np

MOTION_THRESHOLD = 0.01


def calculate_motion(prev_landmarks, curr_landmarks):

    diff = np.abs(
        curr_landmarks - prev_landmarks
    )

    return np.mean(diff)


def is_signing(motion):

    return motion > MOTION_THRESHOLD