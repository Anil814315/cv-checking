"""Training pipeline for attention drift detection.

Expected input CSV columns:
- sequence_id: identifier for contiguous clip/session
- label: binary target (0 = focused, 1 = drift)
- eye_aspect_ratio
- head_orientation_angle
- mouth_opening_ratio
- shoulder_posture_angle

This script builds a sliding window of 30 frames, trains an LSTM model, trains
Random Forest as a baseline, and reports evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

FEATURE_COLUMNS = [
    "eye_aspect_ratio",
    "head_orientation_angle",
    "mouth_opening_ratio",
    "shoulder_posture_angle",
]
WINDOW_SIZE = 30


def build_sequences(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Convert frame-level records into sliding-window sequences.

    Label for each window is the label at the final frame of that window.
    """
    sequences = []
    labels = []

    grouped = df.groupby("sequence_id", sort=False)
    for _, group in grouped:
        group = group.reset_index(drop=True)
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = group["label"].to_numpy(dtype=np.int32)
        if len(group) < window_size:
            continue
        for i in range(len(group) - window_size + 1):
            window_x = values[i : i + window_size]
            window_y = y[i + window_size - 1]
            sequences.append(window_x)
            labels.append(window_y)

    if not sequences:
        raise ValueError("No sequences were created. Check data size and sequence_id grouping.")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def build_lstm_model(input_shape: Tuple[int, int]) -> keras.Model:
    """Create LSTM binary classifier with sigmoid output (drift probability)."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            # ŷ_t = σ(W h_t + b)
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute standard binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train attention drift models (LSTM + RandomForest baseline).")
    parser.add_argument("--data", default="data/attention_features.csv", help="Path to input CSV dataset")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for LSTM")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for LSTM")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = {"sequence_id", "label", *FEATURE_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    x_seq, y = build_sequences(df, window_size=WINDOW_SIZE)

    # Split sequences into train/test.
    x_train, x_test, y_train, y_test = train_test_split(
        x_seq, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features using train frames only (fit on flattened time dimension).
    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    x_test_2d = x_test.reshape(-1, x_test.shape[-1])
    scaler.fit(x_train_2d)

    x_train_scaled = scaler.transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)

    # Train LSTM sequence model.
    lstm = build_lstm_model(input_shape=(WINDOW_SIZE, len(FEATURE_COLUMNS)))
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    lstm.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_prob_lstm = lstm.predict(x_test_scaled, verbose=0).reshape(-1)
    metrics_lstm = evaluate_predictions(y_test, y_prob_lstm)

    # Train Random Forest baseline on flattened sequence windows.
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(x_train_scaled.reshape(x_train_scaled.shape[0], -1), y_train)
    y_prob_rf = rf.predict_proba(x_test_scaled.reshape(x_test_scaled.shape[0], -1))[:, 1]
    metrics_rf = evaluate_predictions(y_test, y_prob_rf)

    # Save artifacts.
    lstm.save(output_dir / "lstm_attention.keras")
    joblib.dump(rf, output_dir / "rf_attention.joblib")
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    metrics = {"lstm": metrics_lstm, "random_forest": metrics_rf}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation Metrics")
    print("------------------")
    for model_name, vals in metrics.items():
        print(f"{model_name}:")
        print(
            "  Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}".format(
                **vals
            )
        )


if __name__ == "__main__":
    # Required on Windows to avoid multiprocessing issues in some contexts.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
