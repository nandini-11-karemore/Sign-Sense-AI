def get_landmark_coordinates(hand_landmarks):
    coordinates = []

    for landmark in hand_landmarks:
        coordinates.append({
            "x": round(landmark.x, 4),
            "y": round(landmark.y, 4),
            "z": round(landmark.z, 4)
        })

    return coordinates