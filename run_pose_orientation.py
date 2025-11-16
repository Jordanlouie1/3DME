"""CLI for running pose orientation on a single image.

Usage:
    python run_pose_orientation.py path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from pose_orientation import estimate_torso_yaw_from_world_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate facing direction from MediaPipe pose world landmarks.")
    parser.add_argument("image", type=Path, help="Path to the RGB image (any format OpenCV can read).")
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=12.0,
        help="Angle in degrees above which the subject is considered left/right facing.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum MediaPipe visibility score required for shoulder landmarks.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="MediaPipe Pose model complexity (2 gives the best world-landmark quality).",
    )
    return parser.parse_args()


def load_image(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def main() -> int:
    args = parse_args()
    image_bgr = load_image(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
    ) as pose:
        result = pose.process(image_rgb)

    if not result.pose_world_landmarks:
        print("Pose world landmarks missing. Ensure the subject is visible and fully in frame.")
        return 1

    orientation = estimate_torso_yaw_from_world_landmarks(
        result.pose_world_landmarks.landmark,
        yaw_threshold_degrees=args.yaw_threshold,
        min_visibility=args.min_visibility,
    )

    print(
        "Image:", args.image,
        "\nFacing:", orientation.facing,
        f"\nYaw: {orientation.yaw_degrees:.2f}°",
        f"\nDepth delta (right-left): {orientation.delta_z:.4f} m",
        f"\nConfidence: {orientation.confidence:.2f}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
