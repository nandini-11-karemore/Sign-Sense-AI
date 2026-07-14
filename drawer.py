import cv2

# Hand landmark connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_hand(frame, landmarks):

    h, w, _ = frame.shape

    # Draw connections
    for start, end in HAND_CONNECTIONS:

        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)

        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Draw landmarks
    for i, lm in enumerate(landmarks):

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(frame, (x,y), 6, (0,0,255), -1)

        cv2.putText(
            frame,
            str(i),
            (x+5,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255,255,255),
            1
        )