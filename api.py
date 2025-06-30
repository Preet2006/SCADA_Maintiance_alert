from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved artifacts
model = joblib.load(os.path.join(BASE_DIR, "future_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "future_scaler.joblib"))
feature_names = joblib.load(os.path.join(BASE_DIR, "future_model_features.joblib"))

app = Flask(__name__)
CORS(app)

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

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input. JSON object expected."}), 400

        # Validate features
        missing = [f for f in feature_names if f not in data]
        extra = [f for f in data if f not in feature_names]

        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        if extra:
            return jsonify({"error": f"Extra/unknown features: {extra}. Please remove them."}), 400

        # Create DataFrame
        input_df = pd.DataFrame([data])[feature_names]

        # Handle NaNs
        if input_df.isnull().any().any():
            return jsonify({"error": "Input contains missing (NaN) values."}), 400

        # Scale and Predict
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
