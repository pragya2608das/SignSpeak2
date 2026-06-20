import tempfile
import streamlit as st

from inference import CTRGCNPredictor
from mediapipe_extractor import extract_video
from gemini_helper import interpret_signs
from speech import speak

from streamlit_webrtc import webrtc_streamer

from webcam_processor import (
    process_frame,
    get_buffer_size,
    get_words,
    clear_words,
    clear_buffer
)
from sign_pipeline import (process_sign)
# -----------------------------------------

# Load Model

# -----------------------------------------

@st.cache_resource
def load_model():
    return CTRGCNPredictor()


predictor = load_model()

# -----------------------------------------

# Page Config

# -----------------------------------------

st.set_page_config(
page_title="SignSpeak 2.0",
page_icon="🤟",
layout="wide"
)

# -----------------------------------------

# Header

# -----------------------------------------

st.title(
"🤟 SignSpeak 2.0"
)

st.markdown(
"""
Real-Time Sign Language Translation Assistant

CTR-GCN + MediaPipe + Gemini + Speech
"""
)

# -----------------------------------------

# Session State

# -----------------------------------------

if "translation" not in st.session_state:
    st.session_state.translation = ""


# -----------------------------------------

# Mode Selection

# -----------------------------------------

mode = st.radio(
"Choose Mode",
[
"Video Upload",
"Live Webcam"
]
)
# ==================================================
# VIDEO UPLOAD
# ==================================================

if mode == "Video Upload":

    uploaded = st.file_uploader(
        "Upload Sign Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded:

        st.video(uploaded)

        # Save uploaded video
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(
                uploaded.read()
            )

            video_path = tmp.name

        # Extract landmarks
        with st.spinner(
            "Extracting landmarks..."
        ):

            sequence = extract_video(
                video_path
            )

        st.success(
            "Landmarks Extracted"
        )

        # Run CTR-GCN
        with st.spinner(
            "Running CTR-GCN..."
        ):

            predictions = predictor.predict(
                sequence
            )

        st.subheader(
            "Top Predictions"
        )

        words = []

        for item in predictions:

            words.append(
                item["word"]
            )

            st.write(
                f"**{item['word']}** "
                f"({item['confidence']:.2f}%)"
            )

        best_word = words[0]

        st.success(
            f"Detected Sign: {best_word}"
        )

        # Translate
        if st.button(
            "Translate"
        ):

            sentence = interpret_signs(
                [best_word]
            )

            st.session_state.translation = (
                sentence
            )

        # Show translation
        if st.session_state.translation:

            st.subheader(
                "Translation"
            )

            st.info(
                st.session_state.translation
            )

            # Speak translation
            if st.button(
                "🔊 Speak Translation"
            ):

                speak(
                    st.session_state.translation
                )




# ==================================================
# WEBCAM
# ==================================================

elif mode == "Live Webcam":

    st.subheader(
        "📷 Live Webcam"
    )

    st.info(
        "Show signs to the webcam."
    )

    webrtc_streamer(
        key="signcam",
        video_frame_callback=process_frame,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    # -----------------------------------------
    # Frame Counter
    # -----------------------------------------

    st.write(
        f"Frames Collected: {get_buffer_size()}/32"
    )

    # -----------------------------------------
    # Predict Sign
    # -----------------------------------------

    if st.button(
        "🔍 Predict Current Sign"
    ):

        word = process_sign(
            predictor
        )

        if word:

            st.success(
                f"Detected: {word}"
            )

        else:

            st.warning(
                "Need at least 32 frames."
            )

    st.divider()

    # -----------------------------------------
    # Detected Signs
    # -----------------------------------------

    st.subheader(
        "Detected Signs"
    )

    detected_words = get_words()

    if len(detected_words) == 0:

        st.warning(
            "No signs detected yet."
        )

    else:

        for word in detected_words:

            st.success(word)

    # -----------------------------------------
    # Buttons
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "Translate Sentence"
        ):

            sentence = interpret_signs(
                detected_words
            )

            st.session_state.translation = (
                sentence
            )

    with col2:

        if st.button(
            "Speak"
        ):

            if st.session_state.translation:

                speak(
                    st.session_state.translation
                )

    with col3:

        if st.button(
            "Clear"
        ):

            clear_words()

            clear_buffer()

            st.session_state.translation = ""

            st.rerun()

    # -----------------------------------------
    # Translation Output
    # -----------------------------------------

    if st.session_state.translation:

        st.divider()

        st.subheader(
            "Translation"
        )

        st.info(
            st.session_state.translation
        )