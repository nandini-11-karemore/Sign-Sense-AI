import streamlit as st # type: ignore
import cv2
import av # type: ignore
import time
from collections import deque, Counter

from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

# -----------------------------
# Import Existing Modules
# -----------------------------
from hand_detector import HandDetector
from drawer import draw_hand
from predict import predict_letter
from ui import draw_dashboard
from word_builder import WordBuilder


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="SignSense AI",
    page_icon="🤟",
    layout="wide",
)

col1, col2 = st.columns([3, 1])

with col1:
    st.title("🤟 SignSense AI")
    st.caption("Real-Time Sign Language Detection using MediaPipe + XGBoost")

with col2:
    st.success("Live Detection")

st.sidebar.title("Controls")
st.sidebar.info(
    """
    Press **Start** below to enable webcam.

    Features:
    - Real-time detection
    - Word Builder
    - Sentence Builder
    - Confidence Score
    """
)

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# -----------------------------
# Video Processor
# -----------------------------
class SignProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = HandDetector()
        self.word_builder = WordBuilder()

        self.prediction_buffer = deque(maxlen=10)

        self.prev_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Flip image for mirror effect
        img = cv2.flip(img, 1)

        prediction = "-"
        confidence = 0
        hands = 0

        # Detect Hands
        result = self.detector.detect(img)

        if result.hand_landmarks:
            hands = len(result.hand_landmarks)

            for hand_landmarks in result.hand_landmarks:
                # Draw skeleton
                draw_hand(img, hand_landmarks)

                # Predict Letter
                letter, conf = predict_letter(hand_landmarks)

                confidence = conf
                self.prediction_buffer.append(letter)

                prediction = Counter(
                    self.prediction_buffer
                ).most_common(1)[0][0]

                # Update Word Builder
                self.word_builder.update(prediction)
        else:
            self.word_builder.reset_prediction()

        # FPS
        current_time = time.time()
        fps = int(1 / max(current_time - self.prev_time, 0.001))
        self.prev_time = current_time

        # Dashboard
        img = draw_dashboard(
            frame=img,
            prediction=prediction,
            confidence=confidence,
            fps=fps,
            progress=self.word_builder.get_progress(),
            word=self.word_builder.get_word(),
            sentence=self.word_builder.get_sentence(),
            hands=hands,
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


# -----------------------------
# Webcam
# -----------------------------
webrtc_streamer(
    key="signsense",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=SignProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
)

st.markdown("---")
st.write("Developed with ❤️ using Streamlit, MediaPipe and XGBoost")