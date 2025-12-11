"""
Flask Backend for AI Signal Classifier
=======================================

This backend loads a Keras LSTM model and StandardScaler to classify
signal data as AIR or METAL.

Setup Instructions:
1. Install dependencies: pip install flask flask-cors tensorflow pandas scikit-learn openpyxl
2. Place model files in the same directory:
   - model.keras (trained Keras LSTM model)
   - scaler.pkl (fitted StandardScaler)
3. Run: python app.py
4. The API will be available at http://localhost:5000

API Endpoints:
- POST /predict: Upload Excel file for classification
- GET /health: Health check endpoint
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# TensorFlow import (adjust based on your installation)
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration - Look for models in parent directory's models folder (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'model.keras'))
SCALER_PATH = os.environ.get('SCALER_PATH', os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
EXPECTED_TIME_STEPS = 684  # Expected number of time steps
EXPECTED_FEATURES = 4  # Current, Voltage, Resistance, Force

# Column indices for signal data (0-indexed)
# Assumes Excel format: Time-Current, Current, Time-Voltage, Voltage, Time-Resistance, Resistance, Time-Force, Force
SIGNAL_COLUMN_INDICES = [1, 3, 5, 7]  # Current, Voltage, Resistance, Force

# Global model and scaler
model = None
scaler = None


def load_model_and_scaler():
    """Load the Keras model and StandardScaler."""
    global model, scaler
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {model.input_shape}")
    
    print(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess Excel data for model prediction.
    
    Args:
        df: DataFrame with 8 columns (4 time columns + 4 signal columns)
    
    Returns:
        Preprocessed numpy array of shape (1, time_steps, 4)
    """
    # Extract signal columns (ignore time columns)
    signal_data = df.iloc[:, SIGNAL_COLUMN_INDICES].values
    
    # Handle NaN values
    signal_data = np.nan_to_num(signal_data, nan=0.0)
    
    # Get current shape
    current_time_steps = signal_data.shape[0]
    
    # Pad or truncate to expected time steps
    if current_time_steps < EXPECTED_TIME_STEPS:
        # Pad with zeros
        padding = np.zeros((EXPECTED_TIME_STEPS - current_time_steps, EXPECTED_FEATURES))
        signal_data = np.vstack([signal_data, padding])
    elif current_time_steps > EXPECTED_TIME_STEPS:
        # Truncate
        signal_data = signal_data[:EXPECTED_TIME_STEPS, :]
    
    # Reshape for scaler: (time_steps * features,) -> scale -> reshape back
    original_shape = signal_data.shape
    signal_flat = signal_data.reshape(-1, EXPECTED_FEATURES)
    
    # Apply scaler
    signal_scaled = scaler.transform(signal_flat)
    
    # Reshape for model: (1, time_steps, features)
    signal_scaled = signal_scaled.reshape(1, EXPECTED_TIME_STEPS, EXPECTED_FEATURES)
    
    return signal_scaled


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict classification from uploaded Excel file.
    
    Expects: multipart/form-data with 'file' field containing Excel file
    Returns: JSON with prediction and confidence
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'File must be an Excel file (.xlsx or .xls)'}), 400
    
    try:
        # Read Excel file
        df = pd.read_excel(file)
        
        # Validate columns
        if df.shape[1] < 8:
            return jsonify({
                'error': f'Expected 8 columns, found {df.shape[1]}'
            }), 400
        
        # Preprocess data
        processed_data = preprocess_data(df)
        
        # Run prediction
        prediction_proba = model.predict(processed_data, verbose=0)
        
        # Get predicted class and confidence
        # Assuming binary classification: 0 = AIR, 1 = METAL
        if prediction_proba.shape[-1] == 1:
            # Sigmoid output
            confidence = float(prediction_proba[0][0])
            predicted_class = "METAL" if confidence > 0.5 else "AIR"
            if predicted_class == "AIR":
                confidence = 1 - confidence
        else:
            # Softmax output (2 classes)
            class_idx = int(np.argmax(prediction_proba[0]))
            confidence = float(prediction_proba[0][class_idx])
            predicted_class = "AIR" if class_idx == 0 else "METAL"
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 4),
            'raw_probabilities': prediction_proba[0].tolist()
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-json', methods=['POST'])
def predict_json():
    """
    Predict classification from JSON data.
    
    Expects: JSON with 'signalData' array of shape (time_steps, 4)
    Returns: JSON with prediction and confidence
    """
    try:
        data = request.get_json()
        
        if 'signalData' not in data:
            return jsonify({'error': 'Missing signalData field'}), 400
        
        signal_data = np.array(data['signalData'])
        
        if signal_data.shape[1] != EXPECTED_FEATURES:
            return jsonify({
                'error': f'Expected {EXPECTED_FEATURES} features, got {signal_data.shape[1]}'
            }), 400
        
        # Pad/truncate to expected time steps
        current_time_steps = signal_data.shape[0]
        if current_time_steps < EXPECTED_TIME_STEPS:
            padding = np.zeros((EXPECTED_TIME_STEPS - current_time_steps, EXPECTED_FEATURES))
            signal_data = np.vstack([signal_data, padding])
        elif current_time_steps > EXPECTED_TIME_STEPS:
            signal_data = signal_data[:EXPECTED_TIME_STEPS, :]
        
        # Scale
        signal_flat = signal_data.reshape(-1, EXPECTED_FEATURES)
        signal_scaled = scaler.transform(signal_flat)
        signal_scaled = signal_scaled.reshape(1, EXPECTED_TIME_STEPS, EXPECTED_FEATURES)
        
        # Predict
        prediction_proba = model.predict(signal_scaled, verbose=0)
        
        if prediction_proba.shape[-1] == 1:
            confidence = float(prediction_proba[0][0])
            predicted_class = "METAL" if confidence > 0.5 else "AIR"
            if predicted_class == "AIR":
                confidence = 1 - confidence
        else:
            class_idx = int(np.argmax(prediction_proba[0]))
            confidence = float(prediction_proba[0][class_idx])
            predicted_class = "AIR" if class_idx == 0 else "METAL"
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 4)
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Try to load model and scaler
    try:
        load_model_and_scaler()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Server starting without model. Place model files in 'models/' directory.")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
