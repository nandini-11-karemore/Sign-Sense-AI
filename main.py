import cv2
import time
from collections import deque, Counter

from hand_detector import HandDetector
from drawer import draw_hand
from predict import predict_letter
from speech import speak
from ui import draw_dashboard
from word_builder  import WordBuilder

# ==========================================
# Initialize
# ==========================================

detector = HandDetector()
word_builder = WordBuilder()

prediction_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev_time = time.time()

# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break
    frame = cv2.resize(frame, (960, 720))
    # Default values
    prediction = "-"
    confidence = 0
    hands = 0

    # Detect Hand
    result = detector.detect(frame)

    if result.hand_landmarks:

        hands = len(result.hand_landmarks)

        for hand_landmarks in result.hand_landmarks:

            # Draw Skeleton
            draw_hand(frame, hand_landmarks)

            # Predict Letter
            letter, confidence = predict_letter(hand_landmarks)

            prediction_buffer.append(letter)

            prediction = Counter(
                prediction_buffer
            ).most_common(1)[0][0]

            # Update Word Builder
            word_builder.update(prediction)

    else:

        word_builder.reset_prediction()

    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    fps = int(1 / (current_time - prev_time))

    prev_time = current_time

    # ======================================
    # Dashboard
    # ======================================

    frame = draw_dashboard(

        frame=frame,

        prediction=prediction,

        confidence=confidence,

        fps=fps,

        progress=word_builder.get_progress(),

        word=word_builder.get_word(),

        sentence=word_builder.get_sentence(),

        hands=hands

    )

    # ======================================
    # Display
    # ======================================

    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    # ------------------------------
    # ENTER -> Speak
    # ------------------------------

    if key == 13:

        text = (
            word_builder.get_sentence()
            + word_builder.get_word()
        ).strip()

        if text != "":
            speak(text)

    # ------------------------------
    # C -> Clear
    # ------------------------------

    elif key == ord("c"):

        word_builder.clear()

    # ------------------------------
    # ESC / Q -> Quit
    # ------------------------------

    elif key == 27 or key == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()