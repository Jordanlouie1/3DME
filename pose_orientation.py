"""Pose orientation utilities leveraging MediaPipe pose world landmarks.

This module lives at the project root so you can keep the upstream MoGe repo
untouched while still reusing the same functionality from your own code base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, Sequence


class LandmarkLike(Protocol):
    """Subset of MediaPipe's landmark attributes used by the helper."""

    x: float
    y: float
    z: float
    visibility: float | None


PoseFacing = Literal["left", "right", "front", "unknown"]


@dataclass
class PoseOrientationResult:
    yaw_degrees: float
    facing: PoseFacing
    confidence: float
    delta_z: float


def estimate_torso_yaw_from_world_landmarks(
    landmarks: Sequence[LandmarkLike] | Iterable[LandmarkLike],
    *,
    left_shoulder_index: int = 11,
    right_shoulder_index: int = 12,
    min_visibility: float = 0.5,
    yaw_threshold_degrees: float = 10.0,
) -> PoseOrientationResult:
    """Return yaw and facing label from MediaPipe pose world landmarks.

    MediaPipe Pose world coordinates follow a right-handed convention with the Z
    axis pointing away from the camera.  When a subject rotates to their left,
    their left shoulder moves closer to the camera (smaller Z).  Comparing that
    depth difference against the shoulder span yields a robust yaw estimate.
    """

    left = _get_landmark(landmarks, left_shoulder_index)
    right = _get_landmark(landmarks, right_shoulder_index)

    _validate_landmark(left, "left_shoulder", min_visibility)
    _validate_landmark(right, "right_shoulder", min_visibility)

    shoulder_span = math.hypot(left.x - right.x, left.y - right.y)
    if shoulder_span < 1e-6:
        raise ValueError("Degenerate shoulder span; cannot determine yaw.")

    delta_z = right.z - left.z  # positive => facing left
    yaw = math.degrees(math.atan2(delta_z, shoulder_span))
    facing = _yaw_to_label(yaw, yaw_threshold_degrees)
    confidence = _compute_confidence(delta_z, shoulder_span, left, right)

    return PoseOrientationResult(
        yaw_degrees=yaw,
        facing=facing,
        confidence=confidence,
        delta_z=delta_z,
    )


def _compute_confidence(
    delta_z: float,
    shoulder_span: float,
    left: LandmarkLike,
    right: LandmarkLike,
) -> float:
    depth_contrast = min(1.0, abs(delta_z) / shoulder_span)
    visibility_left = getattr(left, "visibility", 1.0) or 1.0
    visibility_right = getattr(right, "visibility", 1.0) or 1.0
    confidence = depth_contrast * min(visibility_left, visibility_right)
    return float(max(0.0, min(1.0, confidence)))


def _yaw_to_label(yaw_degrees: float, threshold: float) -> PoseFacing:
    if math.isnan(yaw_degrees):
        return "unknown"
    if yaw_degrees > threshold:
        return "left"
    if yaw_degrees < -threshold:
        return "right"
    return "front"


def _validate_landmark(landmark: LandmarkLike | None, name: str, min_visibility: float) -> None:
    if landmark is None:
        raise ValueError(f"Missing landmark: {name}")

    visibility = getattr(landmark, "visibility", None)
    if visibility is not None and visibility < min_visibility:
        raise ValueError(f"Landmark '{name}' visibility {visibility:.2f} < {min_visibility}")


def _get_landmark(
    landmarks: Sequence[LandmarkLike] | Iterable[LandmarkLike], index: int
) -> LandmarkLike | None:
    if isinstance(landmarks, Sequence):
        if index >= len(landmarks):
            return None
        return landmarks[index]

    for idx, landmark in enumerate(landmarks):
        if idx == index:
            return landmark
    return None


__all__ = [
    "PoseOrientationResult",
    "PoseFacing",
    "estimate_torso_yaw_from_world_landmarks",
]
