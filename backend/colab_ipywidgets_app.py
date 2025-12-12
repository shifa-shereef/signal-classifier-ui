"""
LSTM Signal Classifier - ipywidgets Version for Google Colab
=============================================================
This version uses ipywidgets instead of Tkinter for full Colab compatibility.

Usage:
1. Upload model.keras and scaler.pkl to Colab
2. Run all cells in order
3. Use the interactive widgets to upload Excel and run predictions
"""

# =============================================================================
# INSTALL AND IMPORTS
# =============================================================================

# Uncomment these lines when running in Colab:
# !pip install openpyxl -q

import os
import pickle
import numpy as np
import pandas as pd
from IPython.display import display, HTML, clear_output

# ipywidgets for Colab-native GUI
import ipywidgets as widgets

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "model.keras"  # or "model.h5"
SCALER_PATH = "scaler.pkl"
EXPECTED_TIME_STEPS = 684
EXPECTED_FEATURES = 4

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def load_excel_data(file_content, filename):
    """Load and extract features from Excel file."""
    import io
    
    # Read Excel from bytes
    df = pd.read_excel(io.BytesIO(file_content))
    print(f"✅ Loaded: {filename}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Extract features from columns at indices 1, 3, 5, 7
    # (Current, Voltage, Resistance, Force)
    if df.shape[1] >= 8:
        features = df.iloc[:, [1, 3, 5, 7]].values.astype(np.float32)
    else:
        # Try to extract by column names
        feature_cols = [
            'Current (kA) - Current',
            'Voltage (V) - Voltage', 
            'Resistance (Ohm) - Resistance',
            'Force (kg) - Force'
        ]
        features = df[feature_cols].values.astype(np.float32)
    
    print(f"   Features shape: {features.shape}")
    return features, df


def preprocess_features(features, scaler=None):
    """Preprocess features for LSTM input."""
    # Handle NaN
    features = np.nan_to_num(features, nan=0.0)
    
    # Pad or truncate
    current_steps = features.shape[0]
    if current_steps < EXPECTED_TIME_STEPS:
        padding = np.zeros((EXPECTED_TIME_STEPS - current_steps, EXPECTED_FEATURES))
        features = np.vstack([features, padding])
    elif current_steps > EXPECTED_TIME_STEPS:
        features = features[:EXPECTED_TIME_STEPS, :]
    
    # Apply scaler
    if scaler is not None:
        features_2d = features.reshape(-1, EXPECTED_FEATURES)
        features = scaler.transform(features_2d).reshape(EXPECTED_TIME_STEPS, EXPECTED_FEATURES)
    
    # Reshape for LSTM
    return features.reshape(1, EXPECTED_TIME_STEPS, EXPECTED_FEATURES)


def run_prediction(model, features):
    """Run model prediction."""
    prediction = model.predict(features, verbose=0)
    
    if len(prediction.shape) > 1:
        raw_score = float(prediction[0][0])
    else:
        raw_score = float(prediction[0])
    
    pred_class = "METAL" if raw_score > 0.5 else "AIR"
    confidence = raw_score if raw_score > 0.5 else (1 - raw_score)
    
    return pred_class, confidence, raw_score


# =============================================================================
# IPYWIDGETS GUI CLASS
# =============================================================================

class ColabSignalClassifier:
    """Interactive Signal Classifier for Google Colab using ipywidgets."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.df = None
        
        # Create widgets
        self.create_widgets()
        
        # Load model
        self.load_model_and_scaler()
    
    def create_widgets(self):
        """Create all interactive widgets."""
        
        # Title
        self.title = widgets.HTML(
            value="""
            <div style="text-align: center; padding: 20px;">
                <h1 style="color: #1E40AF; margin-bottom: 5px;">🔬 LSTM Signal Classifier</h1>
                <p style="color: #6B7280; font-size: 14px;">Binary Classification: AIR vs METAL Spot Welds</p>
            </div>
            """
        )
        
        # Status area
        self.model_status = widgets.HTML(
            value='<p style="color: #F59E0B;">⏳ Loading model...</p>'
        )
        self.scaler_status = widgets.HTML(
            value='<p style="color: #F59E0B;">⏳ Loading scaler...</p>'
        )
        
        # File uploader
        self.file_uploader = widgets.FileUpload(
            accept='.xlsx,.xls',
            multiple=False,
            description='Upload Excel',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        self.file_uploader.observe(self.on_file_upload, names='value')
        
        # File status
        self.file_status = widgets.HTML(
            value='<p style="color: #6B7280;">No file uploaded</p>'
        )
        
        # Predict button
        self.predict_btn = widgets.Button(
            description='🚀 Run Prediction',
            button_style='success',
            disabled=True,
            layout=widgets.Layout(width='200px', height='50px'),
            style={'font_weight': 'bold'}
        )
        self.predict_btn.on_click(self.on_predict_click)
        
        # Progress bar
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='',
            bar_style='info',
            layout=widgets.Layout(width='300px', visibility='hidden')
        )
        
        # Results area
        self.result_html = widgets.HTML(
            value="""
            <div style="text-align: center; padding: 30px; background: #F3F4F6; border-radius: 10px; margin-top: 20px;">
                <p style="color: #9CA3AF; font-size: 18px;">Results will appear here</p>
            </div>
            """
        )
        
        # Data preview output
        self.data_preview = widgets.Output()
        
        # Layout
        status_box = widgets.VBox([
            widgets.HTML('<h3>📊 Status</h3>'),
            self.model_status,
            self.scaler_status
        ], layout=widgets.Layout(padding='10px', border='1px solid #E5E7EB', border_radius='8px'))
        
        upload_box = widgets.VBox([
            widgets.HTML('<h3>📁 Data Input</h3>'),
            self.file_uploader,
            self.file_status
        ], layout=widgets.Layout(padding='10px', border='1px solid #E5E7EB', border_radius='8px', margin='10px 0'))
        
        action_box = widgets.VBox([
            self.predict_btn,
            self.progress
        ], layout=widgets.Layout(align_items='center', padding='10px'))
        
        results_box = widgets.VBox([
            widgets.HTML('<h3>🎯 Prediction Results</h3>'),
            self.result_html
        ], layout=widgets.Layout(padding='10px'))
        
        preview_box = widgets.VBox([
            widgets.HTML('<h3>📋 Data Preview</h3>'),
            self.data_preview
        ], layout=widgets.Layout(padding='10px'))
        
        # Main container
        self.main_container = widgets.VBox([
            self.title,
            status_box,
            upload_box,
            action_box,
            results_box,
            preview_box
        ], layout=widgets.Layout(max_width='800px', margin='0 auto'))
    
    def load_model_and_scaler(self):
        """Load model and scaler files."""
        # Load model
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
                self.model_status.value = f'<p style="color: #10B981;">✅ Model loaded: {MODEL_PATH}</p>'
            else:
                self.model_status.value = f'<p style="color: #EF4444;">❌ Model not found: {MODEL_PATH}</p>'
        except Exception as e:
            self.model_status.value = f'<p style="color: #EF4444;">❌ Model error: {str(e)}</p>'
        
        # Load scaler
        try:
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.scaler_status.value = f'<p style="color: #10B981;">✅ Scaler loaded: {SCALER_PATH}</p>'
            else:
                self.scaler_status.value = '<p style="color: #F59E0B;">⚠️ Scaler not found (no normalization)</p>'
        except Exception as e:
            self.scaler_status.value = f'<p style="color: #EF4444;">❌ Scaler error: {str(e)}</p>'
    
    def on_file_upload(self, change):
        """Handle file upload event."""
        if change['new']:
            try:
                # Get uploaded file
                uploaded_file = list(change['new'].values())[0]
                filename = list(change['new'].keys())[0]
                content = uploaded_file['content']
                
                # Process file
                self.features, self.df = load_excel_data(content, filename)
                
                # Update status
                self.file_status.value = f'''
                <p style="color: #10B981;">
                    ✅ <strong>{filename}</strong><br>
                    Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}
                </p>
                '''
                
                # Show preview
                with self.data_preview:
                    clear_output()
                    display(HTML('<p><strong>First 10 rows:</strong></p>'))
                    display(self.df.head(10))
                
                # Enable prediction
                if self.model is not None:
                    self.predict_btn.disabled = False
                    
            except Exception as e:
                self.file_status.value = f'<p style="color: #EF4444;">❌ Error: {str(e)}</p>'
                self.predict_btn.disabled = True
    
    def on_predict_click(self, button):
        """Handle prediction button click."""
        if self.model is None or self.features is None:
            return
        
        # Show progress
        self.progress.layout.visibility = 'visible'
        self.progress.value = 30
        self.predict_btn.disabled = True
        
        try:
            # Preprocess
            self.progress.value = 50
            processed = preprocess_features(self.features, self.scaler)
            
            # Predict
            self.progress.value = 80
            pred_class, confidence, raw_score = run_prediction(self.model, processed)
            
            # Display results
            self.progress.value = 100
            self.display_results(pred_class, confidence, raw_score)
            
        except Exception as e:
            self.result_html.value = f'''
            <div style="text-align: center; padding: 30px; background: #FEE2E2; border-radius: 10px;">
                <p style="color: #EF4444; font-size: 16px;">❌ Error: {str(e)}</p>
            </div>
            '''
        finally:
            self.progress.layout.visibility = 'hidden'
            self.predict_btn.disabled = False
    
    def display_results(self, pred_class, confidence, raw_score):
        """Display prediction results with styling."""
        if pred_class == "AIR":
            color = "#3B82F6"
            emoji = "🔵"
            bg_color = "#DBEAFE"
        else:
            color = "#EF4444"
            emoji = "🔴"
            bg_color = "#FEE2E2"
        
        confidence_pct = int(confidence * 100)
        
        self.result_html.value = f'''
        <div style="text-align: center; padding: 30px; background: {bg_color}; border-radius: 15px; margin-top: 20px;">
            <p style="font-size: 48px; margin: 0;">{emoji}</p>
            <h2 style="color: {color}; font-size: 36px; margin: 10px 0;">{pred_class}</h2>
            
            <div style="margin: 20px auto; max-width: 300px;">
                <p style="color: #374151; margin-bottom: 5px;">Confidence</p>
                <div style="background: #E5E7EB; border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="background: {color}; width: {confidence_pct}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
                </div>
                <p style="color: {color}; font-size: 24px; font-weight: bold; margin-top: 5px;">{confidence_pct}%</p>
            </div>
            
            <p style="color: #6B7280; font-size: 12px; margin-top: 15px;">
                Raw model output: {raw_score:.6f}
            </p>
        </div>
        '''
    
    def display(self):
        """Display the complete interface."""
        display(self.main_container)


# =============================================================================
# MAIN - RUN THIS IN A COLAB CELL
# =============================================================================

def create_classifier():
    """Create and display the classifier interface."""
    print("🔬 Initializing LSTM Signal Classifier...")
    print("-" * 50)
    
    classifier = ColabSignalClassifier()
    classifier.display()
    
    return classifier


# Instructions for Colab
USAGE_INSTRUCTIONS = """
================================================================================
📋 USAGE INSTRUCTIONS FOR GOOGLE COLAB
================================================================================

1. Upload your model files to Colab:
   - model.keras (or model.h5)
   - scaler.pkl

   You can use:
   from google.colab import files
   files.upload()

2. Make sure this script is in the same directory as the model files,
   or update MODEL_PATH and SCALER_PATH at the top of this file.

3. Run all cells, then call:
   
   classifier = create_classifier()

4. Use the upload button to select your Excel file.

5. Click "Run Prediction" to classify the signal.

================================================================================
"""

print(USAGE_INSTRUCTIONS)

# Uncomment the line below to auto-run when the script is executed:
# classifier = create_classifier()
