import cv2
import mediapipe as mp
import streamlit as st

st.write("MediaPipe version:", mp.__version__)
st.write("Has solutions:", hasattr(mp, "solutions"))
st.write(dir(mp))
st.stop()

class DetectionResult:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                hand_landmarks.append(hand.landmark)

        return DetectionResult(hand_landmarks)