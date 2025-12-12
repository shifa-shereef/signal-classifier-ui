"""
LSTM Signal Classifier - Tkinter GUI for Google Colab
======================================================
Binary classification between AIR and METAL spot welds.

Usage in Google Colab:
1. Upload model.keras (or model.h5) and scaler.pkl to Colab
2. Run this script in a code cell
3. The Tkinter window will appear for file selection and prediction

Note: For Colab, you may need to use the local runtime or 
install additional packages for GUI support.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pickle
import numpy as np
import pandas as pd

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install: pip install tensorflow")
    raise

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model and scaler paths - adjust these for your Colab environment
MODEL_PATH = "model.keras"  # or "model.h5"
SCALER_PATH = "scaler.pkl"

# Expected data shape
EXPECTED_TIME_STEPS = 684
EXPECTED_FEATURES = 4

# Column names in the Excel file
EXPECTED_COLUMNS = [
    'Time - Current', 'Current (kA) - Current',
    'Time - Voltage', 'Voltage (V) - Voltage',
    'Time - Resistance', 'Resistance (Ohm) - Resistance',
    'Time - Force', 'Force (kg) - Force'
]

# Feature columns to extract (signal values, not time columns)
FEATURE_COLUMNS = [
    'Current (kA) - Current',
    'Voltage (V) - Voltage',
    'Resistance (Ohm) - Resistance',
    'Force (kg) - Force'
]


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def load_and_validate_excel(file_path):
    """
    Load Excel file and validate its structure.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with the loaded data
        
    Raises:
        ValueError: If file structure is invalid
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded Excel file with shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")
        
        # Check if we have the expected columns
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            # Try alternative column matching (partial match)
            actual_cols = df.columns.tolist()
            feature_indices = [1, 3, 5, 7]  # Columns at indices 1, 3, 5, 7 contain signals
            
            if len(actual_cols) >= 8:
                print("Using positional column extraction (indices 1, 3, 5, 7)")
                return df, feature_indices
            else:
                raise ValueError(f"Missing columns: {missing_cols}")
        
        return df, FEATURE_COLUMNS
        
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {str(e)}")


def extract_features(df, columns_or_indices):
    """
    Extract the 4 feature columns from the DataFrame.
    
    Args:
        df: Input DataFrame
        columns_or_indices: Either column names or indices
        
    Returns:
        numpy array of shape (time_steps, 4)
    """
    if isinstance(columns_or_indices, list) and isinstance(columns_or_indices[0], int):
        # Using indices
        features = df.iloc[:, columns_or_indices].values
    else:
        # Using column names
        features = df[columns_or_indices].values
    
    print(f"Extracted features shape: {features.shape}")
    return features.astype(np.float32)


def preprocess_data(features, scaler=None):
    """
    Preprocess features for LSTM input.
    
    Args:
        features: numpy array of shape (time_steps, 4)
        scaler: Optional StandardScaler for normalization
        
    Returns:
        Preprocessed array ready for model input (1, time_steps, 4)
    """
    # Handle NaN values
    if np.isnan(features).any():
        print("Warning: NaN values found, replacing with 0")
        features = np.nan_to_num(features, nan=0.0)
    
    # Pad or truncate to expected time steps
    current_steps = features.shape[0]
    
    if current_steps < EXPECTED_TIME_STEPS:
        # Pad with zeros
        padding = np.zeros((EXPECTED_TIME_STEPS - current_steps, EXPECTED_FEATURES))
        features = np.vstack([features, padding])
        print(f"Padded from {current_steps} to {EXPECTED_TIME_STEPS} time steps")
    elif current_steps > EXPECTED_TIME_STEPS:
        # Truncate
        features = features[:EXPECTED_TIME_STEPS, :]
        print(f"Truncated from {current_steps} to {EXPECTED_TIME_STEPS} time steps")
    
    # Apply scaler if provided
    if scaler is not None:
        original_shape = features.shape
        features_2d = features.reshape(-1, EXPECTED_FEATURES)
        features_scaled = scaler.transform(features_2d)
        features = features_scaled.reshape(original_shape)
        print("Applied StandardScaler normalization")
    
    # Reshape for LSTM: (batch_size, time_steps, features)
    features = features.reshape(1, EXPECTED_TIME_STEPS, EXPECTED_FEATURES)
    print(f"Final input shape: {features.shape}")
    
    return features


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def load_lstm_model(model_path):
    """
    Load the trained LSTM model.
    
    Args:
        model_path: Path to the saved model (.keras or .h5)
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    return model


def load_scaler(scaler_path):
    """
    Load the fitted StandardScaler.
    
    Args:
        scaler_path: Path to the saved scaler (.pkl)
        
    Returns:
        Loaded scaler or None if not found
    """
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler not found at {scaler_path}, proceeding without normalization")
        return None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Scaler loaded from: {scaler_path}")
    return scaler


def predict(model, features):
    """
    Run prediction on preprocessed features.
    
    Args:
        model: Loaded Keras model
        features: Preprocessed input array (1, time_steps, 4)
        
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Get raw prediction
    prediction = model.predict(features, verbose=0)
    
    # Handle different output shapes
    if len(prediction.shape) > 1:
        confidence = float(prediction[0][0])
    else:
        confidence = float(prediction[0])
    
    # Binary classification: 0 = AIR, 1 = METAL
    # If output is sigmoid (0-1), threshold at 0.5
    predicted_class = "METAL" if confidence > 0.5 else "AIR"
    
    # Adjust confidence for display
    # If AIR (< 0.5), show confidence as 1 - prediction
    display_confidence = confidence if confidence > 0.5 else (1 - confidence)
    
    print(f"Raw prediction: {confidence:.4f}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {display_confidence:.2%}")
    
    return predicted_class, display_confidence, confidence


# =============================================================================
# TKINTER GUI APPLICATION
# =============================================================================

class SignalClassifierApp:
    """
    Tkinter GUI Application for LSTM Signal Classification.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 LSTM Signal Classifier - AIR vs METAL")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Configure style
        self.configure_styles()
        
        # State variables
        self.model = None
        self.scaler = None
        self.excel_path = None
        self.features = None
        
        # Build UI
        self.create_widgets()
        
        # Load model and scaler on startup
        self.load_model_and_scaler()
    
    def configure_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')  # Use 'clam' theme for better appearance
        
        # Configure colors
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 10))
        style.configure('Result.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('AIR.TLabel', foreground='#3B82F6', font=('Helvetica', 24, 'bold'))
        style.configure('METAL.TLabel', foreground='#EF4444', font=('Helvetica', 24, 'bold'))
        
        # Button styles
        style.configure('Action.TButton', font=('Helvetica', 11), padding=10)
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🔬 LSTM Signal Classifier",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Binary Classification: AIR vs METAL Spot Welds",
            style='Status.TLabel'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Model status frame
        status_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.model_status_label = ttk.Label(
            status_frame,
            text="⏳ Loading model...",
            style='Status.TLabel'
        )
        self.model_status_label.pack(anchor=tk.W)
        
        self.scaler_status_label = ttk.Label(
            status_frame,
            text="⏳ Loading scaler...",
            style='Status.TLabel'
        )
        self.scaler_status_label.pack(anchor=tk.W)
        
        # File upload frame
        upload_frame = ttk.LabelFrame(main_frame, text="Data Input", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.upload_btn = ttk.Button(
            upload_frame,
            text="📁 Upload Excel File",
            command=self.upload_file,
            style='Action.TButton'
        )
        self.upload_btn.pack(pady=5)
        
        self.file_label = ttk.Label(
            upload_frame,
            text="No file selected",
            style='Status.TLabel'
        )
        self.file_label.pack(pady=5)
        
        # Prediction button
        self.predict_btn = ttk.Button(
            main_frame,
            text="🚀 Run Prediction",
            command=self.run_prediction,
            style='Action.TButton',
            state=tk.DISABLED
        )
        self.predict_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        self.result_class_label = ttk.Label(
            results_frame,
            text="---",
            style='Result.TLabel'
        )
        self.result_class_label.pack(pady=10)
        
        # Confidence bar
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        
        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            mode='determinate',
            length=200,
            maximum=100
        )
        self.confidence_bar.pack(side=tk.LEFT, padx=10)
        
        self.confidence_label = ttk.Label(
            confidence_frame,
            text="0%",
            style='Status.TLabel'
        )
        self.confidence_label.pack(side=tk.LEFT)
        
        # Raw output
        self.raw_output_label = ttk.Label(
            results_frame,
            text="Raw model output: ---",
            style='Status.TLabel'
        )
        self.raw_output_label.pack(pady=5)
    
    def load_model_and_scaler(self):
        """Load the LSTM model and scaler in a background thread."""
        def load_task():
            try:
                # Load model
                self.model = load_lstm_model(MODEL_PATH)
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"✅ Model loaded: {MODEL_PATH}"
                ))
            except Exception as e:
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"❌ Model error: {str(e)}"
                ))
            
            try:
                # Load scaler
                self.scaler = load_scaler(SCALER_PATH)
                if self.scaler:
                    self.root.after(0, lambda: self.scaler_status_label.config(
                        text=f"✅ Scaler loaded: {SCALER_PATH}"
                    ))
                else:
                    self.root.after(0, lambda: self.scaler_status_label.config(
                        text="⚠️ Scaler not found (proceeding without normalization)"
                    ))
            except Exception as e:
                self.root.after(0, lambda: self.scaler_status_label.config(
                    text=f"❌ Scaler error: {str(e)}"
                ))
        
        thread = threading.Thread(target=load_task)
        thread.daemon = True
        thread.start()
    
    def upload_file(self):
        """Handle Excel file upload."""
        filetypes = [
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=filetypes
        )
        
        if file_path:
            try:
                self.excel_path = file_path
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"📄 {filename}")
                
                # Load and validate
                df, columns = load_and_validate_excel(file_path)
                self.features = extract_features(df, columns)
                
                # Enable prediction button
                if self.model is not None:
                    self.predict_btn.config(state=tk.NORMAL)
                
                messagebox.showinfo(
                    "Success",
                    f"File loaded successfully!\n\n"
                    f"Rows: {df.shape[0]}\n"
                    f"Columns: {df.shape[1]}\n"
                    f"Features extracted: {self.features.shape}"
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.file_label.config(text="No file selected")
                self.predict_btn.config(state=tk.DISABLED)
    
    def run_prediction(self):
        """Run the LSTM prediction."""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        if self.features is None:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Start progress
        self.progress.start()
        self.predict_btn.config(state=tk.DISABLED)
        
        def prediction_task():
            try:
                # Preprocess
                processed = preprocess_data(self.features, self.scaler)
                
                # Predict
                pred_class, confidence, raw = predict(self.model, processed)
                
                # Update UI
                self.root.after(0, lambda: self.display_results(pred_class, confidence, raw))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Prediction Error",
                    f"Failed to run prediction:\n{str(e)}"
                ))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=prediction_task)
        thread.daemon = True
        thread.start()
    
    def display_results(self, pred_class, confidence, raw):
        """Display prediction results in the UI."""
        # Set class label with appropriate color
        if pred_class == "AIR":
            self.result_class_label.config(
                text=f"🔵 {pred_class}",
                style='AIR.TLabel'
            )
        else:
            self.result_class_label.config(
                text=f"🔴 {pred_class}",
                style='METAL.TLabel'
            )
        
        # Update confidence bar
        confidence_pct = int(confidence * 100)
        self.confidence_bar['value'] = confidence_pct
        self.confidence_label.config(text=f"{confidence_pct}%")
        
        # Raw output
        self.raw_output_label.config(text=f"Raw model output: {raw:.6f}")


# =============================================================================
# COLAB-SPECIFIC ENTRY POINT
# =============================================================================

def run_in_colab():
    """
    Run the Tkinter app in Google Colab.
    
    For Colab, you have several options:
    1. Use local runtime connected to your machine
    2. Use ngrok or similar for remote display
    3. Use the newer Colab features for GUI support
    """
    print("=" * 60)
    print("LSTM Signal Classifier - Tkinter GUI")
    print("=" * 60)
    print()
    print("Starting Tkinter application...")
    print()
    print("Note: For Google Colab, ensure you're using local runtime")
    print("or have X11 forwarding configured.")
    print()
    
    root = tk.Tk()
    app = SignalClassifierApp(root)
    root.mainloop()


def main():
    """Standard entry point for running the application."""
    run_in_colab()


# =============================================================================
# COLAB NOTEBOOK CELL CODE
# =============================================================================

COLAB_INSTRUCTIONS = """
================================================================================
HOW TO RUN IN GOOGLE COLAB
================================================================================

Option 1: Local Runtime (Recommended)
--------------------------------------
1. Install Colab local runtime: pip install jupyter_http_over_ws
2. Enable extension: jupyter serverextension enable --py jupyter_http_over_ws
3. Start local runtime: jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com'
4. Connect Colab to local runtime
5. Run this script

Option 2: Use Colab with display forwarding
--------------------------------------------
# In a Colab cell, run:

# Install required packages
!pip install openpyxl

# For X11 forwarding (requires setup)
# !apt-get install -y xvfb
# !pip install pyvirtualdisplay

# Then run the app
# import colab_gui_app
# colab_gui_app.main()

Option 3: Alternative - Use ipywidgets for Colab
-------------------------------------------------
If Tkinter doesn't work, consider using ipywidgets which are native to Colab.
See the ipywidgets version of this classifier below.

================================================================================
"""

print(COLAB_INSTRUCTIONS)


if __name__ == "__main__":
    main()
