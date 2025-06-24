from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

# ✅ Get the current directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load saved artifacts using relative paths
model = joblib.load(os.path.join(BASE_DIR, "future_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "future_scaler.joblib"))
feature_names = joblib.load(os.path.join(BASE_DIR, "future_model_features.joblib"))

app = Flask(__name__)
CORS(app)  # optional if calling from browser

@app.route("/", methods=["GET"])
def home():
    return "✅ Predictive Maintenance API is up and running."

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # 🧪 Validate that all required features are present
        missing = [f for f in feature_names if f not in data]
        extra = [f for f in data if f not in feature_names]

        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        if extra:
            return jsonify({"error": f"Extra/unknown features present: {extra}. Please remove them."}), 400

        # 🐍 Prepare input
        input_df = pd.DataFrame([data])[feature_names]

        # ⚖️ Scale the input
        scaled_input = scaler.transform(input_df)

        # 🎯 Prediction
        prediction = model.predict(scaled_input)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
