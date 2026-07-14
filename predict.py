import joblib
import pandas as pd

model = joblib.load("models/alphabet_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

FEATURE_NAMES = []

for i in range(21):
    FEATURE_NAMES.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

def predict_letter(hand_landmarks):

    row = []

    for lm in hand_landmarks:
        row.extend([lm.x, lm.y, lm.z])

    df = pd.DataFrame([row], columns=FEATURE_NAMES)

    # Prediction
    prediction = model.predict(df)[0]

    # Probability
    probabilities = model.predict_proba(df)[0]

    confidence = max(probabilities) * 100

    letter = encoder.inverse_transform([prediction])[0]

    return letter, confidence