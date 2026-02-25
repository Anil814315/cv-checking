"""Live webcam attention drift prediction.

Runs real-time feature extraction and sequence inference using the trained LSTM.
Displays drift score on the OpenCV window with color-coded status:
- Focused (Green)
- Moderate Drift (Yellow)
- High Drift (Red)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from tensorflow import keras

from feature_extraction import AttentionFeatureExtractor

# Keep local constants so this module is standalone.
WINDOW_SIZE = 30
N_FEATURES = 4


# BGR colors for OpenCV display.
GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)


def status_from_score(score: float) -> tuple[str, tuple[int, int, int]]:
    """Map drift score to text status + color."""
    if score < 0.33:
        return "Focused", GREEN
    if score < 0.66:
        return "Moderate Drift", YELLOW
    return "High Drift", RED


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live attention drift prediction from webcam.")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained artifacts")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (Windows default is 0)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    lstm = keras.models.load_model(model_dir / "lstm_attention.keras")
    rf = joblib.load(model_dir / "rf_attention.joblib")
    scaler = joblib.load(model_dir / "feature_scaler.joblib")

    extractor = AttentionFeatureExtractor()
    feature_window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    score_window: deque[float] = deque(maxlen=WINDOW_SIZE)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW improves Windows camera startup.
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            features = extractor.extract_features(frame)
            if features is not None:
                feature_window.append(features.to_array())

            current_score = 0.0
            rf_score = 0.0
            if len(feature_window) == WINDOW_SIZE:
                seq = np.array(feature_window, dtype=np.float32)
                seq_scaled = scaler.transform(seq).reshape(1, WINDOW_SIZE, N_FEATURES)

                # LSTM sigmoid output: ŷ_t in [0, 1]
                y_hat_t = float(lstm.predict(seq_scaled, verbose=0)[0][0])
                score_window.append(y_hat_t)

                # DriftScore = (1/T) Σ ŷ_t (moving average over latest T predictions).
                current_score = float(np.mean(score_window))

                # Baseline model output for comparison/debugging.
                rf_score = float(
                    rf.predict_proba(seq_scaled.reshape(1, WINDOW_SIZE * N_FEATURES))[0][1]
                )

            status_text, color = status_from_score(current_score)

            # Draw overlay panel.
            cv2.rectangle(frame, (10, 10), (460, 135), (20, 20, 20), thickness=-1)
            cv2.putText(
                frame,
                f"Drift Score: {current_score:.3f}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"RF Baseline Score: {rf_score:.3f}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Attention Drift Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
