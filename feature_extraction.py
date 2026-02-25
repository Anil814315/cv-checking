"""Feature extraction utilities for attention drift detection.

This module uses MediaPipe Holistic landmarks + OpenCV frames to compute
frame-level features used by sequence models:
1) Eye aspect ratio (eye openness)
2) Head orientation angle
3) Mouth opening ratio
4) Shoulder posture angle
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# Face mesh landmark indices for common facial geometry calculations.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
NOSE_TIP_IDX = 1


@dataclass
class FrameFeatures:
    """Container for computed features in one frame."""

    eye_aspect_ratio: float
    head_orientation_angle: float
    mouth_opening_ratio: float
    shoulder_posture_angle: float

    def to_array(self) -> np.ndarray:
        """Return feature vector in fixed order expected by models."""
        return np.array(
            [
                self.eye_aspect_ratio,
                self.head_orientation_angle,
                self.mouth_opening_ratio,
                self.shoulder_posture_angle,
            ],
            dtype=np.float32,
        )


class AttentionFeatureExtractor:
    """Extracts face/body features from webcam frames using MediaPipe."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._mp_holistic = mp.solutions.holistic
        self.holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        self.holistic.close()

    @staticmethod
    def _distance(a, b) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _angle_degrees(a, b) -> float:
        """Angle in degrees for vector a->b in image coordinates."""
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def _eye_aspect_ratio(self, face_lm, width: int, height: int, eye_idx: list[int]) -> float:
        """Compute Eye Aspect Ratio (EAR) from 6 eye landmarks."""
        pts = [(face_lm[i].x * width, face_lm[i].y * height) for i in eye_idx]
        vertical_1 = self._distance(pts[1], pts[5])
        vertical_2 = self._distance(pts[2], pts[4])
        horizontal = max(self._distance(pts[0], pts[3]), 1e-6)
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def extract_features(self, frame_bgr: np.ndarray) -> Optional[FrameFeatures]:
        """Extract frame-level features from a BGR image.

        Returns:
            FrameFeatures if all required landmarks are present, else None.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        if results.face_landmarks is None or results.pose_landmarks is None:
            return None

        h, w = frame_bgr.shape[:2]
        face_lm = results.face_landmarks.landmark
        pose_lm = results.pose_landmarks.landmark

        # 1) Eye openness (EAR): average both eyes.
        left_ear = self._eye_aspect_ratio(face_lm, w, h, LEFT_EYE_IDX)
        right_ear = self._eye_aspect_ratio(face_lm, w, h, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0

        # 2) Head orientation angle: roll-like angle from eye line.
        left_eye_outer = (face_lm[LEFT_EYE_IDX[0]].x * w, face_lm[LEFT_EYE_IDX[0]].y * h)
        right_eye_outer = (face_lm[RIGHT_EYE_IDX[3]].x * w, face_lm[RIGHT_EYE_IDX[3]].y * h)
        head_angle = self._angle_degrees(left_eye_outer, right_eye_outer)

        # 3) Mouth opening ratio: lip distance normalized by mouth width.
        upper_lip = (face_lm[MOUTH_UPPER_IDX].x * w, face_lm[MOUTH_UPPER_IDX].y * h)
        lower_lip = (face_lm[MOUTH_LOWER_IDX].x * w, face_lm[MOUTH_LOWER_IDX].y * h)
        mouth_left = (face_lm[MOUTH_LEFT_IDX].x * w, face_lm[MOUTH_LEFT_IDX].y * h)
        mouth_right = (face_lm[MOUTH_RIGHT_IDX].x * w, face_lm[MOUTH_RIGHT_IDX].y * h)
        mouth_open = self._distance(upper_lip, lower_lip)
        mouth_width = max(self._distance(mouth_left, mouth_right), 1e-6)
        mouth_ratio = mouth_open / mouth_width

        # 4) Shoulder posture angle: slope of left->right shoulder line.
        left_shoulder = pose_lm[self._mp_holistic.PoseLandmark.LEFT_SHOULDER].x * w, pose_lm[
            self._mp_holistic.PoseLandmark.LEFT_SHOULDER
        ].y * h
        right_shoulder = pose_lm[self._mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * w, pose_lm[
            self._mp_holistic.PoseLandmark.RIGHT_SHOULDER
        ].y * h
        shoulder_angle = self._angle_degrees(left_shoulder, right_shoulder)

        return FrameFeatures(
            eye_aspect_ratio=ear,
            head_orientation_angle=head_angle,
            mouth_opening_ratio=mouth_ratio,
            shoulder_posture_angle=shoulder_angle,
        )
