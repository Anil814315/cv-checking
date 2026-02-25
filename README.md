# Attention Drift Detection (Desktop, Python)

A modular desktop system that detects **attention drift** from webcam video using:
- **OpenCV** for video capture and live UI
- **MediaPipe Holistic** for face + upper-body landmarks
- **TensorFlow/Keras LSTM** for temporal drift detection
- **Random Forest** as baseline model

## Project Structure

- `feature_extraction.py` - Extracts frame-level features from landmarks.
- `train_model.py` - Training pipeline for LSTM + Random Forest, plus evaluation metrics.
- `live_prediction.py` - Live webcam inference and drift score visualization.
- `requirements.txt` - Python dependencies.

## Features Computed per Frame

1. **Eye Aspect Ratio (EAR)** - eye openness.
2. **Head Orientation Angle** - eye-line orientation angle.
3. **Mouth Opening Ratio** - lip gap normalized by mouth width.
4. **Shoulder Posture Angle** - shoulder-line tilt angle.

These are stored in a **sliding window of 30 frames** and fed to sequence models.

## Model Outputs

- LSTM produces sigmoid probability:
  - `ŷ_t = σ(W h_t + b)`
- Drift score is moving average:
  - `DriftScore = (1/T) Σ ŷ_t`

## Data Format (CSV)

Create `data/attention_features.csv` with columns:

- `sequence_id` (int/string)
- `label` (0 focused, 1 drift)
- `eye_aspect_ratio`
- `head_orientation_angle`
- `mouth_opening_ratio`
- `shoulder_posture_angle`

Each `sequence_id` should represent a continuous clip/session of frames.

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train_model.py --data data/attention_features.csv --output-dir models --epochs 20
```

Saved artifacts:
- `models/lstm_attention.keras`
- `models/rf_attention.joblib`
- `models/feature_scaler.joblib`
- `models/metrics.json`

Metrics reported:
- Accuracy
- Precision
- Recall
- F1-score

## Live Prediction

```bash
python live_prediction.py --model-dir models --camera-index 0
```

Press `q` to quit.

UI status colors:
- **Focused** (Green)
- **Moderate Drift** (Yellow)
- **High Drift** (Red)

## Notes

- Ensure webcam permission is enabled in Windows Privacy Settings.
- If your camera index differs, try `--camera-index 1` or `2`.
