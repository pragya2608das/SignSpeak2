import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from mediapipe_extractor import extract_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Last 32 frames
sequence_buffer = deque(maxlen=32)

# Detected words
recognized_words = []

# -----------------------------------------
# Process Webcam Frame
# -----------------------------------------

def process_frame(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = holistic.process(rgb)

    landmarks = extract_landmarks(
        results
    )

    sequence_buffer.append(
        landmarks
    )

    # Draw landmarks
    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:

        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:

        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    return image


# -----------------------------------------
# Get 32-frame sequence
# -----------------------------------------

def get_sequence():

    if len(sequence_buffer) < 32:

        return None

    return list(sequence_buffer)


# -----------------------------------------
# Buffer Info
# -----------------------------------------

def get_buffer_size():

    return len(sequence_buffer)


# -----------------------------------------
# Words
# -----------------------------------------

def add_word(word):

    recognized_words.append(word)


def get_words():

    return recognized_words


def clear_words():

    recognized_words.clear()


# -----------------------------------------
# Clear
# -----------------------------------------

def clear_buffer():

    sequence_buffer.clear()