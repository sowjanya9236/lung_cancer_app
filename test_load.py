import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.load(os.path.join(BASE_DIR, "model", "knn_model.pkl"))
joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

print("Loaded successfully")
