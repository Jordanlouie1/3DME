"""Batch-extract pose orientation + 3D landmark coordinates from images.

MediaPipe Pose provides world landmarks (in meters, Z away from camera).
This helper runs the pose detector on every image in a directory and stores
the yaw/facing output from :mod:`pose_orientation` along with XYZ values for
selected landmarks.  The resulting JSON file can be fed into
``stitch_ply_icp.py --metadata`` to prime the ICP alignment.

Example:

```
python extract_pose_metadata.py \
    --images-dir images \
    --output yaw_map.json \
    --landmarks left_shoulder,right_shoulder,left_hip,right_hip
```
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import mediapipe as mp

from pose_orientation import estimate_torso_yaw_from_world_landmarks


LANDMARK_NAME_TO_INDEX: Dict[str, int] = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing RGB images (jpg/png/etc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pose_metadata.json"),
        help="Destination JSON mapping image -> pose metadata",
    )
    parser.add_argument(
        "--landmarks",
        type=str,
        default="left_shoulder,right_shoulder,left_hip,right_hip",
        help="Comma-separated list of Mediapipe landmark names to record",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="MediaPipe Pose model complexity (2 has best world landmark quality)",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=12.0,
        help="Facing threshold passed to pose_orientation (degrees)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum shoulder visibility required for reliable yaw",
    )
    return parser.parse_args()


def resolve_landmarks(arg: str) -> List[str]:
    names = [token.strip().lower() for token in arg.split(",") if token.strip()]
    for name in names:
        if name not in LANDMARK_NAME_TO_INDEX:
            raise SystemExit(f"Unknown landmark: {name}. Pick from {list(LANDMARK_NAME_TO_INDEX)}")
    return names


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def extract_landmark_dict(landmarks: Iterable, names: Iterable[str]) -> Dict[str, dict | None]:
    result: Dict[str, dict | None] = {}
    for name in names:
        idx = LANDMARK_NAME_TO_INDEX[name]
        if idx >= len(landmarks):
            result[name] = None
            continue
        lm = landmarks[idx]
        if lm is None:
            result[name] = None
            continue
        result[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(getattr(lm, "visibility", 0.0) or 0.0),
        }
    return result


def average_xyz(entries: Iterable[dict | None]) -> dict | None:
    coords = [e for e in entries if e is not None]
    if not coords:
        return None
    avg = {
        axis: float(sum(c[axis] for c in coords) / len(coords))
        for axis in ("x", "y", "z")
    }
    return avg


def main() -> None:
    args = parse_args()
    names = resolve_landmarks(args.landmarks)

    if not args.images_dir.exists():
        raise SystemExit(f"Image directory {args.images_dir} does not exist")

    image_paths = sorted(
        [p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    mp_pose = mp.solutions.pose
    metadata: Dict[str, dict] = {}

    with mp_pose.Pose(static_image_mode=True, model_complexity=args.model_complexity) as pose:
        for path in image_paths:
            image_bgr = load_image(path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if not result.pose_world_landmarks:
                metadata[path.name] = {"error": "pose_world_landmarks missing"}
                continue

            world_landmarks = result.pose_world_landmarks.landmark
            orientation = estimate_torso_yaw_from_world_landmarks(
                world_landmarks,
                yaw_threshold_degrees=args.yaw_threshold,
                min_visibility=args.min_visibility,
            )

            selected = extract_landmark_dict(world_landmarks, names)
            torso_center = average_xyz(selected.values())

            metadata[path.name] = {
                "yaw_degrees": orientation.yaw_degrees,
                "facing": orientation.facing,
                "confidence": orientation.confidence,
                "delta_z": orientation.delta_z,
                "landmarks": selected,
                "torso_center": torso_center,
            }
            print(
                f"Processed {path.name}: yaw={orientation.yaw_degrees:.2f}°, "
                f"facing={orientation.facing}, confidence={orientation.confidence:.2f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Saved metadata for {len(metadata)} images to {args.output}")


if __name__ == "__main__":
    main()

