# AI Signal Classifier - Python Backend

This Flask backend provides the prediction API for the AI Signal Classifier.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Place Model Files

Create a `models/` directory and place your trained model files:

```
backend/
├── models/
│   ├── model.keras      # Your trained Keras LSTM model
│   └── scaler.pkl       # Fitted StandardScaler
├── app.py
├── requirements.txt
└── README.md
```

### 3. Run the Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /predict

Upload an Excel file for classification.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` - Excel file (.xlsx or .xls)

**Response:**
```json
{
  "prediction": "AIR",
  "confidence": 0.9234,
  "raw_probabilities": [0.9234, 0.0766]
}
```

### POST /predict-json

Send signal data as JSON for classification.

**Request:**
```json
{
  "signalData": [
    [1.23, 4.56, 0.01, 100.5],
    [1.24, 4.57, 0.01, 101.2],
    ...
  ]
}
```

**Response:**
```json
{
  "prediction": "METAL",
  "confidence": 0.8756
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## Excel File Format

The Excel file should have 8 columns in this order:

| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 | Column 7 | Column 8 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| Time - Current | Current (kA) | Time - Voltage | Voltage (V) | Time - Resistance | Resistance (Ohm) | Time - Force | Force (kg) |

The backend extracts columns 2, 4, 6, 8 (the signal values) and ignores the time columns.

## Model Requirements

Your model should:
- Accept input shape: `(batch_size, 684, 4)`
- Output: Binary classification (AIR vs METAL)
  - Either sigmoid (1 output) or softmax (2 outputs)

The scaler should be a fitted `sklearn.preprocessing.StandardScaler`.

## Deployment

### Using Gunicorn (Production)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Environment Variables

- `PORT`: Server port (default: 5000)
- `MODEL_PATH`: Path to model file (default: models/model.keras)
- `SCALER_PATH`: Path to scaler file (default: models/scaler.pkl)

## Connecting Frontend

Update the frontend to call your backend API:

```typescript
const API_URL = 'http://localhost:5000';

async function classifySignal(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}
```

## Troubleshooting

**Model not loading:**
- Ensure `model.keras` is a valid Keras model file
- Check TensorFlow version compatibility

**Scaler errors:**
- Ensure `scaler.pkl` was pickled with compatible scikit-learn version
- The scaler should be fitted on 4 features

**Input shape mismatch:**
- The model expects exactly 684 time steps and 4 features
- Data is automatically padded/truncated to match
