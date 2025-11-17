"""Stitch multiple PLY scans together using orientation hints + ICP.

The script expects one or more `.ply` files inside an input directory
(default: `MogeProcessed/`).  Supply a JSON metadata file produced from the
pose orientation pipeline to seed the yaw (rotation about the up-axis) for
each scan.  A good initial yaw drastically improves ICP convergence.

Example metadata structure (degrees):

```
{
  "front.ply": 0,
  "left.ply": 90,
  "right.ply": -90,
  "back.ply": 180
}
```

Usage:

```
python stitch_ply_icp.py \
    --input-dir "MogeProcessed" \
    --metadata yaw_map.json \
    --output stitched_model.ply
```

Install dependencies via `pip install -r requirements.txt` (Open3D + NumPy).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans


@dataclass
class PoseHint:
    yaw_degrees: float | None = None
    torso_center: tuple[float, float, float] | None = None
    manual_rotation: tuple[float, float, float] | None = None  # Euler XYZ degrees
    manual_translation: tuple[float, float, float] | None = None


@dataclass
class CloudTask:
    path: Path
    pose_hint: PoseHint | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("MogeProcessed"),
        help="Directory containing .ply scans",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="JSON file mapping scan filename to yaw degrees (from pose orientation)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stitched.ply"),
        help="Destination path for the merged point cloud",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Down-sampling voxel size in meters (set 0 to disable)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.02,
        help="Max correspondence distance for ICP in meters",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=200,
        help="Maximum ICP iterations per pair",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=0,
        help="If >1, run KMeans clustering and keep the largest cluster before ICP",
    )
    parser.add_argument(
        "--plane-distance",
        type=float,
        default=0.0,
        help="If >0, remove the dominant plane via RANSAC with this distance threshold",
    )
    parser.add_argument(
        "--plane-ransac-n",
        type=int,
        default=3,
        help="Number of points sampled for each RANSAC plane hypothesis",
    )
    parser.add_argument(
        "--plane-iterations",
        type=int,
        default=200,
        help="RANSAC iterations when removing the dominant plane",
    )
    parser.add_argument(
        "--use-global-registration",
        action="store_true",
        help="If set, perform FPFH-based global registration before ICP",
    )
    parser.add_argument(
        "--global-voxel",
        type=float,
        default=0.02,
        help="Voxel size for feature extraction when global registration is enabled",
    )
    parser.add_argument(
        "--global-ransac-distance",
        type=float,
        default=0.03,
        help="Max correspondence distance during global RANSAC registration",
    )
    parser.add_argument(
        "--global-ransac-iterations",
        type=int,
        default=400000,
        help="Number of RANSAC iterations for global registration",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("camera_config.json"),
        help="Optional JSON mapping filename patterns to manual transforms",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="Comma-separated list specifying the order to add scans (filenames or stems)",
    )
    parser.add_argument(
        "--export-intermediate",
        action="store_true",
        help="If set, write combined cloud after each merge to output_<name>.ply",
    )
    return parser.parse_args()


def load_pose_hints(path: Path | None) -> Dict[str, PoseHint]:
    hints: Dict[str, PoseHint] = {}
    if path is None:
        return hints
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    def add_entry(key: str, value) -> None:
        hint = _build_pose_hint(value)
        hints[_normalize_key(key)] = hint

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            add_entry(key, value)
        return hints
    if isinstance(raw, Iterable):
        for item in raw:
            if not isinstance(item, Mapping) or "file" not in item:
                raise ValueError("List metadata entries must be objects with a 'file' key")
            add_entry(str(item["file"]), item)
        return hints
    raise ValueError("Unsupported metadata JSON structure")


def load_camera_config(path: Path | None) -> Dict[str, PoseHint]:
    config: Dict[str, PoseHint] = {}
    if path is None or not path.exists():
        return config
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    def add_entry(key: str, value) -> None:
        config[_normalize_key(key)] = _build_pose_hint(value)

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            add_entry(key, value)
        return config
    if isinstance(raw, Iterable):
        for item in raw:
            if not isinstance(item, Mapping) or "file" not in item:
                raise ValueError("Camera config entries must include a 'file' key")
            add_entry(str(item["file"]), item)
        return config
    raise ValueError("Unsupported camera config structure")


def _extract_yaw(value) -> float:
    if isinstance(value, Mapping):
        if "yaw" in value:
            return float(value["yaw"])
        if "yaw_degrees" in value:
            return float(value["yaw_degrees"])
    return float(value)


def _extract_center(value) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping) and "torso_center" in value:
        center = value["torso_center"]
        if isinstance(center, Mapping):
            try:
                return (
                    float(center["x"]),
                    float(center["y"]),
                    float(center["z"]),
                )
            except KeyError:
                return None
    if isinstance(value, Mapping) and "center" in value:
        center = value["center"]
        if isinstance(center, Mapping):
            try:
                return (
                    float(center["x"]),
                    float(center["y"]),
                    float(center["z"]),
                )
            except KeyError:
                return None
    return None


def _build_pose_hint(value) -> PoseHint:
    yaw = None
    try:
        yaw = _extract_yaw(value)
    except (TypeError, ValueError):
        yaw = None
    center = _extract_center(value)
    manual_rotation = _extract_rotation(value)
    manual_translation = _extract_vector(value, "translation")
    return PoseHint(
        yaw_degrees=yaw,
        torso_center=center,
        manual_rotation=manual_rotation,
        manual_translation=manual_translation,
    )


def _extract_rotation(value) -> tuple[float, float, float] | None:
    for key in ("rotation_degrees", "rotation", "euler_degrees"):
        vec = _extract_vector(value, key)
        if vec is not None:
            return vec
    return None


def _extract_vector(value, key: str) -> tuple[float, float, float] | None:
    if not isinstance(value, Mapping) or key not in value:
        return None
    raw = value[key]
    if isinstance(raw, Mapping):
        components = [raw.get(axis) for axis in ("x", "y", "z")]
    else:
        components = raw
    try:
        arr = [float(component) for component in components]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid vector for '{key}': {raw}") from exc
    if len(arr) != 3:
        raise ValueError(f"Vector '{key}' must have 3 elements, got {arr}")
    return tuple(arr)  # type: ignore[return-value]


def _normalize_key(name: str) -> str:
    return Path(name).stem.lower()


def infer_default_yaw(name: str) -> float | None:
    lowered = name.lower()
    if "front" in lowered:
        return 0.0
    if "back" in lowered:
        return 180.0
    if "left" in lowered:
        return 90.0
    if "right" in lowered:
        return -90.0
    if "top" in lowered or "up" in lowered:
        return None
    return None


def merge_pose_hints(base: PoseHint | None, override: PoseHint | None) -> PoseHint | None:
    if base is None and override is None:
        return None
    result = PoseHint(
        yaw_degrees=base.yaw_degrees if base else None,
        torso_center=base.torso_center if base else None,
        manual_rotation=base.manual_rotation if base else None,
        manual_translation=base.manual_translation if base else None,
    )
    if override is None:
        return result
    if override.yaw_degrees is not None:
        result.yaw_degrees = override.yaw_degrees
    if override.torso_center is not None:
        result.torso_center = override.torso_center
    if override.manual_rotation is not None:
        result.manual_rotation = override.manual_rotation
    if override.manual_translation is not None:
        result.manual_translation = override.manual_translation
    return result


def yaw_to_transform(yaw_degrees: float) -> np.ndarray:
    # Rotate around +Y (up) axis; invert sign to bring scan back to canonical frame.
    yaw_rad = math.radians(-yaw_degrees)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )
    return transform


def euler_xyz_matrix(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (math.radians(d) for d in degrees_xyz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # Apply X then Y then Z rotations
    return rot_z @ rot_y @ rot_x


def preprocess_cloud(path: Path, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud {path} is empty")
    voxel_size = args.voxel_size
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if args.kmeans_clusters and args.kmeans_clusters > 1:
        cloud = filter_largest_cluster(cloud, args.kmeans_clusters)
    if args.plane_distance > 0:
        cloud = remove_dominant_plane(
            cloud,
            distance_threshold=args.plane_distance,
            ransac_n=args.plane_ransac_n,
            num_iterations=args.plane_iterations,
        )
    search_radius = 2.5 * voxel_size if voxel_size > 0 else 0.02
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
    )
    return cloud


def compute_fpfh_features(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.pipelines.registration.Feature:
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    cloud_down = cloud.voxel_down_sample(voxel_size)
    cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return cloud_down, fpfh


def filter_largest_cluster(cloud: o3d.geometry.PointCloud, k: int) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    if len(points) < k or k < 2:
        return cloud
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(points)
    counts = np.bincount(labels)
    keep_label = int(counts.argmax())
    indices = np.flatnonzero(labels == keep_label)
    if len(indices) == len(points):
        return cloud
    filtered = cloud.select_by_index(indices)
    print(f"KMeans retained {len(indices)} / {len(points)} points (cluster {keep_label})")
    return filtered


def remove_dominant_plane(
    cloud: o3d.geometry.PointCloud,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> o3d.geometry.PointCloud:
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if not inliers:
        return cloud
    retained = cloud.select_by_index(inliers, invert=True)
    print(
        "Removed plane"
        f" (points={len(inliers)}, model={plane_model[0]:.2f}x + {plane_model[1]:.2f}y + {plane_model[2]:.2f}z + {plane_model[3]:.2f})"
    )
    return retained


def stitch_point_clouds(tasks: Iterable[CloudTask], args: argparse.Namespace) -> o3d.geometry.PointCloud:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("No point clouds found to stitch.")

    base_cloud = preprocess_cloud(tasks[0].path, args)
    apply_pose_hint_transform(base_cloud, tasks[0].pose_hint)
    combined = base_cloud

    icp = o3d.pipelines.registration
    criteria = icp.ICPConvergenceCriteria(max_iteration=args.max_iteration)
    estimation = icp.TransformationEstimationPointToPlane()

    for task in tasks[1:]:
        source = preprocess_cloud(task.path, args)
        apply_pose_hint_transform(source, task.pose_hint)

        init = np.eye(4)
        init[:3, 3] = _center_translation(source, combined)
        if args.use_global_registration:
            init = run_global_registration(
                source,
                combined,
                args.global_voxel,
                args.global_ransac_distance,
                args.global_ransac_iterations,
            )
        result = icp.registration_icp(
            source,
            combined,
            args.distance_threshold,
            init,
            estimation,
            criteria,
        )
        print(
            f"Aligned {task.path.name}: fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.4f}"
        )
        source.transform(result.transformation)
        combined += source
        if args.export_intermediate:
            intermediate_path = args.output.with_name(f"{args.output.stem}_{task.path.stem}.ply")
            o3d.io.write_point_cloud(str(intermediate_path), combined)
            print(f"Exported intermediate combined cloud to {intermediate_path}")

    return combined


def _center_translation(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)
    if src_pts.size == 0 or tgt_pts.size == 0:
        return np.zeros(3)
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)
    return tgt_center - src_center


def apply_pose_hint_transform(cloud: o3d.geometry.PointCloud, hint: PoseHint | None) -> None:
    if hint is None:
        return
    if hint.yaw_degrees is not None:
        cloud.transform(yaw_to_transform(hint.yaw_degrees))
    if hint.torso_center is not None:
        transform = np.eye(4)
        transform[:3, 3] = -np.asarray(hint.torso_center)
        cloud.transform(transform)
    if hint.manual_rotation is not None or hint.manual_translation is not None:
        manual = np.eye(4)
        if hint.manual_rotation is not None:
            manual[:3, :3] = euler_xyz_matrix(hint.manual_rotation)
        if hint.manual_translation is not None:
            manual[:3, 3] = np.asarray(hint.manual_translation)
        cloud.transform(manual)


def run_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    distance_threshold: float,
    ransac_iterations: int,
) -> np.ndarray:
    src_down, src_fpfh = compute_fpfh_features(source, voxel_size)
    tgt_down, tgt_fpfh = compute_fpfh_features(target, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, 500),
    )
    print(
        f"Global registration: fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.4f}"
    )
    return result.transformation


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist")

    pose_hints = load_pose_hints(args.metadata)
    manual_hints = load_camera_config(args.camera_config)
    ply_paths = sorted(input_dir.glob("*.ply"))
    if args.order:
        order_tokens = [token.strip() for token in args.order.split(",") if token.strip()]
        prioritized: list[Path] = []
        remaining = ply_paths.copy()
        for token in order_tokens:
            normalized = _normalize_key(token)
            for path in ply_paths:
                if _normalize_key(path.name) == normalized and path not in prioritized:
                    prioritized.append(path)
                    if path in remaining:
                        remaining.remove(path)
                    break
        ply_paths = prioritized + remaining
    tasks: list[CloudTask] = []
    for p in ply_paths:
        key = _normalize_key(p.name)
        hint = merge_pose_hints(pose_hints.get(key), manual_hints.get(key))
        inferred_yaw = None
        if hint is None or hint.yaw_degrees is None:
            inferred_yaw = infer_default_yaw(p.name)
            if inferred_yaw is not None:
                print(f"Inferred yaw {inferred_yaw:.1f}° for {p.name} from filename")
            hint = merge_pose_hints(hint, PoseHint(yaw_degrees=inferred_yaw))
        tasks.append(CloudTask(path=p, pose_hint=hint))

    print(f"Found {len(tasks)} scans. Starting stitching ...")
    merged = stitch_point_clouds(tasks, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.output), merged)
    print(f"Merged point cloud saved to {args.output}")


if __name__ == "__main__":
    main()
