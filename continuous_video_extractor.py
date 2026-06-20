import cv2
import mediapipe as mp
import numpy as np

from mediapipe_extractor import extract_landmarks

mp_holistic = mp.solutions.holistic


def extract_full_video(video_path):

    sequence = []

    cap = cv2.VideoCapture(video_path)

    print("Video:", video_path)
    print("Opened:", cap.isOpened())

    if not cap.isOpened():
        return []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            landmarks = extract_landmarks(
                results
            )

            sequence.append(
                landmarks
            )

    cap.release()

    return np.array(
        sequence,
        dtype=np.float32
    )