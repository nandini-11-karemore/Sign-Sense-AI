import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("dataset/landmarks.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)
print(df.head())

# ==========================================
# Remove classes with fewer than 2 samples
# (Fixes the 'nothing' class issue)
# ==========================================
counts = df["label"].value_counts()

valid_labels = counts[counts >= 2].index

df = df[df["label"].isin(valid_labels)]

print("\nClasses Used for Training:")
print(df["label"].value_counts())

# ==========================================
# Features & Labels
# ==========================================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==========================================
# Encode Labels
# ==========================================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nModel Classes:")
print(encoder.classes_)
print(f"\nTotal Classes: {len(encoder.classes_)}")

# ==========================================
# Train / Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Create Model
# ==========================================
model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

print("\nTraining Model...\n")

# ==========================================
# Train
# ==========================================
model.fit(X_train, y_train)

# ==========================================
# Evaluate
# ==========================================
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print(f"Accuracy : {accuracy * 100:.2f}%")
print("=" * 50)

# ==========================================
# Save Model
# ==========================================
joblib.dump(model, "models/alphabet_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel Saved Successfully!")
print("Model File      : models/alphabet_model.pkl")
print("Label Encoder   : models/label_encoder.pkl")