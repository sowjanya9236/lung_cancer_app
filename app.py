from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "knn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            # Collect all input values
            features = [int(request.form[x]) for x in [
                'gender', 'age', 'smoking', 'yellow_fingers', 'anxiety',
                'peer_pressure', 'chronic_disease', 'fatigue', 'allergy',
                'wheezing', 'alcohol', 'coughing', 'breath_shortness',
                'swallowing_difficulty', 'chest_pain'
            ]]
            features = np.array([features])
            features = scaler.transform(features)

            prediction = model.predict(features)[0]
            result = "LUNG CANCER DETECTED" if prediction == 1 else "NO LUNG CANCER"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)

