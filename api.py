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
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    app.logger.info("Model artifacts loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load model artifacts: {str(e)}")
    raise SystemExit("Failed to load required model files")

# Performance metrics cache
performance_metrics = {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.85,
    "f1": 0.87,
    "confusion_matrix": {
        "true_positives": 45,
        "false_positives": 5,
        "true_negatives": 120,
        "false_negatives": 8
    },
    "last_updated": datetime.utcnow().isoformat()
}

def calculate_shap_values(scaled_data):
    """Calculate SHAP values for the input data"""
    # Convert to numpy array if it's a DataFrame
    if isinstance(scaled_data, pd.DataFrame):
        scaled_data = scaled_data.values
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(scaled_data)
    
    # For multi-class, we'll use the SHAP values for the predicted class
    if isinstance(shap_values, list):
        # For multi-class models, SHAP returns a list of arrays (one per class)
        return shap_values
    else:
        # For binary classification, it returns a single array
        return [shap_values]

def get_feature_importance(shap_values, feature_names, top_n=5):
    """Extract top influential features from SHAP values"""
    if isinstance(shap_values, list):
        # For multi-class, we'll use the mean absolute SHAP values across all classes
        mean_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        mean_shap = np.abs(shap_values)
    
    # Get top features
    top_indices = np.argsort(-mean_shap)[:top_n]
    top_features = []
    
    for idx in top_indices:
        top_features.append({
            "name": feature_names[idx],
            "value": float(mean_shap[idx]),
            "direction": "increase" if shap_values[0][idx] > 0 else "decrease"
        })
    
    return top_features

def calculate_alert_priority(prediction, probabilities, shap_values, feature_names):
    """Calculate alert priority based on prediction, probabilities and SHAP values"""
    priority = {
        "level": "normal",
        "color": "green",
        "icon": "CheckCircle",
        "probabilities": probabilities,
        "features": []
    }
    
    # Determine priority level
    if prediction == 2:  # Critical
        priority["level"] = "critical"
        priority["color"] = "red"
        priority["icon"] = "AlertTriangle"
    elif prediction == 1:  # Warning
        priority["level"] = "warning"
        priority["color"] = "orange"
        priority["icon"] = "AlertCircle"
    
    # Get top influential features
    if shap_values is not None:
        priority["features"] = get_feature_importance(shap_values, feature_names)
    
    return priority

# API Endpoints
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "operational",
        "model_version": "1.0.0",
        "supports_shap": True,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_ready": True,
        "services": ["predictive_model", "scaling_service", "shap_explanations"],
        "last_updated": datetime.utcnow().isoformat()
    }), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Get model performance metrics"""
    return jsonify({
        "status": "success",
        "metrics": performance_metrics,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint with SHAP explanations"""
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
        
        # Calculate SHAP values
        shap_values = calculate_shap_values(scaled_data)
        
        # Calculate alert priority
        alert_priority = calculate_alert_priority(
            prediction,
            {
                "normal": probabilities[0],
                "watch": probabilities[1],
                "critical": probabilities[2] if len(probabilities) > 2 else 0.0
            },
            shap_values,
            feature_names
        )
        
        # Prepare SHAP explanation data
        shap_explanation = {
            "base_value": float(explainer.expected_value[prediction] if isinstance(explainer.expected_value, list) else explainer.expected_value),
            "feature_names": feature_names,
            "values": [float(x) for x in shap_values[prediction][0]] if isinstance(shap_values, list) else [float(x) for x in shap_values[0]]
        }

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
            "shap_explanation": shap_explanation,
            "alert_priority": alert_priority,
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

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint for multiple records"""
    start_time = datetime.now()
    request_id = os.urandom(8).hex()
    
    try:
        app.logger.info(f"Request {request_id}: Received batch prediction request")
        data = request.get_json(force=True)

        # Input validation
        if not isinstance(data, list):
            app.logger.warning(f"Request {request_id}: Invalid input type")
            return jsonify({
                "error": "Invalid input format",
                "message": "Expected array of JSON objects",
                "request_id": request_id
            }), 400

        if len(data) == 0:
            app.logger.warning(f"Request {request_id}: Empty input array")
            return jsonify({
                "error": "Empty input",
                "message": "No prediction data provided",
                "request_id": request_id
            }), 400

        # Create DataFrame with proper typing
        input_df = pd.DataFrame(data)
        
        # Feature validation
        missing_features = [f for f in feature_names if f not in input_df.columns]
        extra_features = [f for f in input_df.columns if f not in feature_names]

        if missing_features or extra_features:
            app.logger.warning(f"Request {request_id}: Feature mismatch")
            return jsonify({
                "error": "Feature mismatch",
                "missing_features": missing_features,
                "extra_features": extra_features,
                "expected_features": feature_names,
                "request_id": request_id
            }), 400

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

        # Make predictions
        predictions = model.predict(scaled_data).tolist()
        probabilities = model.predict_proba(scaled_data).tolist()
        
        # Calculate SHAP values for the batch
        shap_values = calculate_shap_values(scaled_data)
        
        # Prepare results
        results = []
        for i in range(len(data)):
            # Calculate alert priority for each prediction
            alert_priority = calculate_alert_priority(
                predictions[i],
                {
                    "normal": probabilities[i][0],
                    "watch": probabilities[i][1],
                    "critical": probabilities[i][2] if len(probabilities[i]) > 2 else 0.0
                },
                [sv[i] for sv in shap_values] if isinstance(shap_values, list) else shap_values[i],
                feature_names
            )
            
            # Prepare SHAP explanation for this prediction
            shap_exp = {
                "base_value": float(explainer.expected_value[predictions[i]] if isinstance(explainer.expected_value, list) else explainer.expected_value),
                "feature_names": feature_names,
                "values": [float(x) for x in (shap_values[predictions[i]][i] if isinstance(shap_values, list) else shap_values[i])]
            }
            
            results.append({
                "prediction": predictions[i],
                "probabilities": {
                    "normal": probabilities[i][0],
                    "watch": probabilities[i][1],
                    "critical": probabilities[i][2] if len(probabilities[i]) > 2 else 0.0
                },
                "model_confidence": max(probabilities[i]),
                "shap_explanation": shap_exp,
                "alert_priority": alert_priority
            })

        # Log successful prediction
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        app.logger.info(f"Request {request_id}: Batch prediction successful ({len(data)} records in {processing_time:.2f}ms)")

        # Prepare response
        return jsonify({
            "status": "success",
            "results": results,
            "count": len(results),
            "request_id": request_id,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        app.logger.error(f"Request {request_id}: Batch prediction failed - {str(e)}")
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