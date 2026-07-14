import os
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===========================
# PATHS
# ===========================
MODEL_PATH = "models/hand_landmarker.task"
DATASET_PATH = "datasets/asl_alphabet_train/asl_alphabet_train"

# Process only these folders first
LABELS = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

# ===========================
# LOAD MODEL
# ===========================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# ===========================
# CREATE CSV HEADER
# ===========================
columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

rows = []

total_images = 0
detected_images = 0
skipped_images = 0

# ===========================
# PROCESS DATASET
# ===========================

for label in LABELS:

    folder = os.path.join(DATASET_PATH, label)

    print(f"\nProcessing Folder : {label}")

    images = os.listdir(folder)

    for image_name in tqdm(images):

        total_images += 1

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            skipped_images += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if not result.hand_landmarks:
            skipped_images += 1
            continue

        detected_images += 1

        hand = result.hand_landmarks[0]

        row = []

        for lm in hand:
            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        row.append(label)

        rows.append(row)

# ===========================
# SAVE CSV
# ===========================

os.makedirs("dataset", exist_ok=True)

df = pd.DataFrame(rows, columns=columns)

df.to_csv("dataset/landmarks.csv", index=False)

# ===========================
# REPORT
# ===========================

print("\n==============================")
print("Dataset Extraction Complete")
print("==============================")
print(f"Total Images   : {total_images}")
print(f"Detected Hands : {detected_images}")
print(f"Skipped        : {skipped_images}")
print(f"CSV Samples    : {len(df)}")
print("==============================")
print("Saved -> dataset/landmarks.csv")