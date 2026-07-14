import cv2


def draw_dashboard(
    frame,
    prediction,
    confidence,
    fps,
    progress,
    word,
    sentence,
    hands,
):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    dashboard_height = 140

    # Background
    cv2.rectangle(
        overlay,
        (0, height - dashboard_height),
        (width, height),
        (25, 25, 25),
        -1,
    )

    # Transparency
    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
    )

    y = height - 105

    # ---------------- TITLE ----------------

    cv2.putText(
        frame,
        "Sign Language Detector",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # ---------------- Prediction ----------------

    cv2.putText(
        frame,
        f"Prediction : {prediction}",
        (20, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # ---------------- Confidence ----------------

    cv2.putText(
        frame,
        f"Confidence : {confidence:.1f}%",
        (250, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Confidence Bar

    cv2.rectangle(frame, (250, y + 45), (430, y + 60), (255,255,255), 2)

    cv2.rectangle(
        frame,
        (250, y + 45),
        (250 + int(confidence * 1.8), y + 60),
        (0,255,0),
        -1
    )

    # Hold Bar

    cv2.putText(
        frame,
        "Hold",
        (470, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
    )

    cv2.rectangle(frame,(470,y+45),(650,y+60),(255,255,255),2)

    cv2.rectangle(
        frame,
        (470,y+45),
        (470+progress,y+60),
        (255,180,0),
        -1
    )

    # FPS

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (700,y+35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2,
    )

    # Hands

    cv2.putText(
        frame,
        f"Hands : {hands}",
        (700,y+65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2,
    )

    # Word

    cv2.putText(
        frame,
        f"Word : {word}",
        (20,height-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    # Sentence

    cv2.putText(
        frame,
        f"Sentence : {sentence}",
        (350,height-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    return frame
