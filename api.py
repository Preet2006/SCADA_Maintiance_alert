from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import sys

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "future_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "future_scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "future_model_features.joblib")
LOG_PATH = os.path.join(BASE_DIR, "api.log")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging with Unicode support
class UnicodeSafeRotatingFileHandler(RotatingFileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.encode('utf-8').decode('utf-8') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging
handler = UnicodeSafeRotatingFileHandler(LOG_PATH, maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Add console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
app.logger.addHandler(console_handler)

# Load artifacts with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    app.logger.info("Model artifacts loaded successfully")  # Removed Unicode symbol
except Exception as e:
    app.logger.error(f"Failed to load model artifacts: {str(e)}")
    raise SystemExit("Failed to load required model files")

# API Endpoints
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "operational",
        "model_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_ready": True,
        "services": ["predictive_model", "scaling_service"],
        "last_updated": datetime.utcnow().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    start_time = datetime.now()
    request_id = os.urandom(8).hex()
    
    try:
        app.logger.info(f"Request {request_id}: Received prediction request")
        data = request.get_json(force=True)

        # Input validation
        if not isinstance(data, dict):
            app.logger.warning(f"Request {request_id}: Invalid input type")
            return jsonify({
                "error": "Invalid input format",
                "message": "Expected JSON object",
                "request_id": request_id
            }), 400

        # Feature validation
        missing_features = [f for f in feature_names if f not in data]
        extra_features = [f for f in data if f not in feature_names]

        if missing_features or extra_features:
            app.logger.warning(f"Request {request_id}: Feature mismatch")
            return jsonify({
                "error": "Feature mismatch",
                "missing_features": missing_features,
                "extra_features": extra_features,
                "expected_features": feature_names,
                "request_id": request_id
            }), 400

        # Create DataFrame with proper typing
        input_df = pd.DataFrame([data])[feature_names]
        
        # Handle categorical features
        categorical_cols = [col for col in feature_names if col == 'SHIFT']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str)

        # Data quality checks
        if input_df.isnull().any().any():
            app.logger.warning(f"Request {request_id}: Missing values detected")
            return jsonify({
                "error": "Data quality issue",
                "message": "Input contains missing values",
                "request_id": request_id
            }), 400

        # Scale features
        numerical_cols = [col for col in feature_names if col not in categorical_cols]
        scaled_data = input_df.copy()
        scaled_data[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make prediction
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0].tolist()
        
        # Log successful prediction
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        app.logger.info(f"Request {request_id}: Prediction successful in {processing_time:.2f}ms")

        # Prepare response
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probabilities": {
                "normal": probabilities[0],
                "watch": probabilities[1],
                "critical": probabilities[2] if len(probabilities) > 2 else 0.0
            },
            "model_confidence": max(probabilities),
            "request_id": request_id,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        app.logger.error(f"Request {request_id}: Prediction failed - {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"Starting API server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)