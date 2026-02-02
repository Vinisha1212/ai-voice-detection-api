import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from audio_features import extract_audio_features

X = []
y = []

def load_data(folder, label):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".mp3"):
                path = os.path.join(root, file)
                try:
                    features = extract_audio_features(path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print("Error processing:", path, e)


load_data("data/human", 0)
load_data("data/ai", 1)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

#  Pipeline: Scaling + Classifier
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

model.fit(X, y)

joblib.dump(model, "voice_detector.pkl")
print("Model trained and saved successfully.")
