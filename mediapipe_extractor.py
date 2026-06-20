import numpy as np
import cv2
import mediapipe as mp


def extract_landmarks(results):

    pose = np.zeros((33,3), dtype=np.float32)
    left = np.zeros((21,3), dtype=np.float32)
    right = np.zeros((21,3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)

    if results.left_hand_landmarks:
        left = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ], dtype=np.float32)

    if results.right_hand_landmarks:
        right = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ], dtype=np.float32)

    return np.concatenate(
        [pose, left, right],
        axis=0
    )


mp_holistic = mp.solutions.holistic


def extract_video(video_path):

    TARGET_FRAMES = 32

    sequence = []

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        return []

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        TARGET_FRAMES,
        dtype=int
    )

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            success, frame = cap.read()

            if not success:

                sequence.append(
                    np.zeros(
                        (75, 3),
                        dtype=np.float32
                    )
                )

                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(
                rgb
            )

            landmarks = extract_landmarks(
                results
            )

            sequence.append(
                landmarks
            )

    cap.release()

    return sequence