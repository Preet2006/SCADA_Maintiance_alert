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
import shap

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "future_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "future_scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "future_model_features.joblib")
LOG_PATH = os.path.join(BASE_DIR, "api.log")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
handler = RotatingFileHandler(LOG_PATH, maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
app.logger.addHandler(console_handler)

# Load artifacts
try:
    model_config = joblib.load(MODEL_PATH)
    model = model_config['model']
    model_use_encoded = model_config['use_encoded']
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    
    # Load label encoders if they exist
    label_encoders_path = os.path.join(BASE_DIR, "label_encoders.joblib")
    label_encoders = {}
    if os.path.exists(label_encoders_path):
        label_encoders = joblib.load(label_encoders_path)
    
    explainer = shap.TreeExplainer(model)
    app.logger.info("Model artifacts loaded successfully")
    app.logger.info(f"Model classes: {model.classes_}")
    app.logger.info(f"Model uses encoded data: {model_use_encoded}")
    app.logger.info(f"Explainer expected value shape: {np.array(explainer.expected_value).shape}")
except Exception as e:
    app.logger.error(f"Failed to load model artifacts: {str(e)}")
    raise SystemExit("Failed to load required model files")

def get_shap_values(scaled_data):
    """Safe SHAP values calculation with proper dimension handling"""
    try:
        if isinstance(scaled_data, pd.DataFrame):
            scaled_data = scaled_data.values
        
        # For single prediction, ensure 2D array
        if len(scaled_data.shape) == 1:
            scaled_data = scaled_data.reshape(1, -1)
        
        # Calculate SHAP values
        raw_shap = explainer.shap_values(scaled_data)
        app.logger.info(f"Raw SHAP type: {type(raw_shap)}")
        
        # Handle different SHAP output formats
        if isinstance(raw_shap, list):
            # Multi-class case
            app.logger.info(f"SHAP list length: {len(raw_shap)}")
            return [np.array(v) for v in raw_shap]
        elif len(raw_shap.shape) == 3:
            # Multi-class with different shape
            return [raw_shap[i] for i in range(raw_shap.shape[0])]
        else:
            # Binary case
            return [np.array(raw_shap)]
            
    except Exception as e:
        app.logger.error(f"SHAP calculation error: {str(e)}", exc_info=True)
        return None

def prepare_shap_explanation(shap_values, prediction, feature_names):
    """Prepare SHAP explanation with proper array handling"""
    try:
        if shap_values is None:
            return None
        
        # Get base value
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            base_value = float(expected_value[prediction])
        elif hasattr(expected_value, '__len__'):
            base_value = float(expected_value[prediction])
        else:
            base_value = float(expected_value)
        
        # Get SHAP values for this prediction
        if isinstance(shap_values, list):
            # Multi-class case
            values = shap_values[prediction][0].tolist()
        else:
            # Binary case
            values = shap_values[0].tolist()
        
        return {
            "base_value": base_value,
            "feature_names": feature_names,
            "values": [float(x) for x in values]
        }
    except Exception as e:
        app.logger.error(f"SHAP explanation error: {str(e)}", exc_info=True)
        return None

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    start_time = datetime.now()
    request_id = os.urandom(8).hex()
    
    try:
        app.logger.info(f"Request {request_id}: Started")
        data = request.get_json()
        
        if not data:
            raise ValueError("Empty request body")
        
        # Validate features
        missing = [f for f in feature_names if f not in data]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Create DataFrame
        input_df = pd.DataFrame([data])[feature_names]
        
        # Handle categoricals
        categorical_cols = ['SHIFT']  # Add other categorical columns if needed
        for col in categorical_cols:
            if col in feature_names:
                input_df[col] = input_df[col].astype(str)
        
        # Scale features
        numerical_cols = [col for col in feature_names if col not in categorical_cols]
        scaled_data = input_df.copy()
        scaled_data[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Encode categorical features if model requires it
        if model_use_encoded:
            for col in categorical_cols:
                if col in feature_names and col in label_encoders:
                    scaled_data[col] = label_encoders[col].transform(scaled_data[col])
        
        # Make prediction
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0].tolist()
        
        # Get SHAP values
        shap_values = get_shap_values(scaled_data)
        shap_explanation = prepare_shap_explanation(shap_values, prediction, feature_names)
        
        # Prepare response
        response = {
            "status": "success",
            "prediction": prediction,
            "probabilities": {
                "normal": float(probabilities[0]),
                "watch": float(probabilities[1]),
                "critical": float(probabilities[2]) if len(probabilities) > 2 else 0.0
            },
            "shap_explanation": shap_explanation,
            "request_id": request_id,
            "processing_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        app.logger.info(f"Request {request_id}: Success")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Request {request_id}: Failed - {str(e)}", exc_info=True)
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_ready": True,
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)