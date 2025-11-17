"""Align raw scans to a reference 3D model using Open3D + trimesh.

The script converts the provided 3D model (GLB/OBJ/etc.) into a dense point
cloud, then registers every scan in the input directory to that reference via
FPFH-based global registration followed by ICP refinement.  Each aligned scan
is exported, along with a merged point cloud and the rigid transforms in JSON
form for downstream use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import open3d as o3d
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("3Dmodel/ASSET_1763331733_8814326.glb"))
    parser.add_argument("--input-dir", type=Path, default=Path("MogeProcessed"))
    parser.add_argument("--output", type=Path, default=Path("aligned_to_model.ply"))
    parser.add_argument("--transforms", type=Path, default=Path("aligned_transforms.json"))
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("aligned_scans"),
        help="Directory to store individual aligned scans",
    )
    parser.add_argument("--model-samples", type=int, default=400000)
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Down-sample size for scans")
    parser.add_argument(
        "--global-distance",
        type=float,
        default=0.08,
        help="Max correspondence distance for global (RANSAC) registration",
    )
    parser.add_argument(
        "--icp-distance",
        type=float,
        default=0.02,
        help="Max correspondence distance for ICP refinement",
    )
    parser.add_argument(
        "--max-icp-iter",
        type=int,
        default=200,
        help="Maximum ICP iterations",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated list of scan stems to include (e.g., 'left1,right1')",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON metadata providing yaw/rotation hints per scan",
    )
    parser.add_argument(
        "--use-metadata-translation",
        action="store_true",
        help="Apply translation vectors from metadata (e.g., torso_center/translation)",
    )
    parser.add_argument(
        "--manual-translate",
        nargs="*",
        default=None,
        help="Manual translation overrides per scan, e.g. left1=-0.05,0,0 right1=0.05,0,0",
    )
    return parser.parse_args()


@dataclass
class PoseHint:
    yaw_degrees: float | None = None
    translation: Optional[np.ndarray] = None


def load_pose_hints(path: Path | None) -> Dict[str, PoseHint]:
    hints: Dict[str, PoseHint] = {}
    if path is None or not path.exists():
        return hints
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    def add_entry(name: str, value) -> None:
        normalized = _normalize_key(name)
        hints[normalized] = PoseHint(
            yaw_degrees=_extract_yaw(value),
            translation=_extract_translation(value),
        )

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            add_entry(key, value)
    elif isinstance(raw, Iterable):
        for item in raw:
            if isinstance(item, Mapping) and "file" in item:
                add_entry(str(item["file"]), item)
    return hints


def parse_manual_translations(entries: Optional[list[str]]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    if not entries:
        return result
    for entry in entries:
        if "=" not in entry:
            continue
        name, vector_str = entry.split("=", 1)
        components = vector_str.split(",")
        if len(components) != 3:
            continue
        try:
            vector = np.array([float(c) for c in components], dtype=float)
        except ValueError:
            continue
        result[_normalize_key(name)] = vector
    return result


def _extract_yaw(value) -> float | None:
    if isinstance(value, Mapping):
        if "yaw_degrees" in value:
            return float(value["yaw_degrees"])
        if "yaw" in value:
            return float(value["yaw"])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_translation(value) -> Optional[np.ndarray]:
    if not isinstance(value, Mapping):
        return None
    vector = None
    for key in ("torso_center", "center", "translation"):
        if key in value:
            candidate = value[key]
            if isinstance(candidate, Mapping):
                components = [candidate.get(axis) for axis in ("x", "y", "z")]
            else:
                components = candidate
            try:
                vector = np.array([float(component) for component in components], dtype=float)
            except Exception:
                vector = None
            break
    if vector is not None and vector.shape == (3,):
        return vector
    return None


def _normalize_key(name: str) -> str:
    return Path(name).stem.lower()


def infer_default_rotation(name: str) -> float:
    lowered = name.lower()
    if "right" in lowered or "back" in lowered:
        return 180.0
    if "left" in lowered or "front" in lowered:
        return 0.0
    if "top" in lowered:
        return -90.0
    return 0.0


def rotate_point_cloud_in_place(pcd: o3d.geometry.PointCloud, yaw_degrees: float) -> None:
    if abs(yaw_degrees) < 1e-6:
        return
    yaw_rad = np.deg2rad(yaw_degrees)
    # axis-angle vector about +Y
    axis_angle = np.array([0.0, yaw_rad, 0.0])
    R = pcd.get_rotation_matrix_from_axis_angle(axis_angle)
    center = pcd.get_center()
    pcd.translate(-center)
    pcd.rotate(R, center=(0.0, 0.0, 0.0))
    pcd.translate(center)


def load_model_point_cloud(path: Path, sample_count: int) -> o3d.geometry.PointCloud:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene '{path}' contains no geometry")
        mesh = trimesh.util.concatenate(tuple(mesh.dump().values()))
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Model '{path}' is empty or invalid")

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.compute_vertex_normals()
    point_count = min(sample_count, len(mesh.vertices) * 4)
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=point_count, init_factor=5)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.5 if voxel_size > 0 else 0.05
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5 if voxel_size > 0 else 0.1
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd, fpfh


def global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    distance_threshold: float,
) -> np.ndarray:
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
    )
    return result.transformation


def refine_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init: np.ndarray,
    distance_threshold: float,
    max_iteration: int,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )


def align_scans(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} not found")
    model_pc = load_model_point_cloud(args.model, args.model_samples)
    model_down, model_fpfh = preprocess_point_cloud(model_pc, args.voxel_size)

    scan_paths = sorted(args.input_dir.glob("*.ply"))
    if not scan_paths:
        raise SystemExit(f"No .ply scans found in {args.input_dir}")

    include_set: set[str] | None = None
    if args.include:
        tokens = [token.strip().lower() for token in args.include.split(",") if token.strip()]
        include_set = set(tokens) if tokens else None

    args.export_dir.mkdir(parents=True, exist_ok=True)
    transforms: Dict[str, list[list[float]]] = {}
    aligned_clouds: list[o3d.geometry.PointCloud] = []
    pose_hints = load_pose_hints(args.metadata)
    manual_translations = parse_manual_translations(args.manual_translate)

    for scan_path in scan_paths:
        normalized_name = scan_path.stem.lower()
        if include_set and normalized_name not in include_set:
            print(f"Skipping {scan_path.name} (not in include list)")
            continue
        print(f"Aligning {scan_path.name}…")
        scan = o3d.io.read_point_cloud(str(scan_path))
        if scan.is_empty():
            print(f"  Skipping {scan_path.name}: empty scan")
            continue
        hint = pose_hints.get(normalized_name)
        rotation = None
        if hint and hint.yaw_degrees is not None:
            rotation = hint.yaw_degrees
        else:
            rotation = infer_default_rotation(normalized_name)
        if abs(rotation) > 1e-6:
            print(f"  Applying pre-rotation {rotation:.1f}° around Y")
            rotate_point_cloud_in_place(scan, rotation)
        manual_vector = manual_translations.get(normalized_name)
        if args.use_metadata_translation and hint and hint.translation is not None:
            print(f"  Applying metadata translation {-hint.translation}")
            scan.translate(-hint.translation)
        if manual_vector is not None:
            print(f"  Applying manual translation {manual_vector}")
            scan.translate(manual_vector)
        scan_down, scan_fpfh = preprocess_point_cloud(scan, args.voxel_size)
        init = global_registration(
            scan_down,
            model_down,
            scan_fpfh,
            model_fpfh,
            args.global_distance,
        )
        refine = refine_with_icp(scan_down, model_down, init, args.icp_distance, args.max_icp_iter)
        final_transform = refine.transformation
        print(
            f"  fitness={refine.fitness:.3f}, rmse={refine.inlier_rmse:.4f}, det={np.linalg.det(final_transform[:3,:3]):.3f}"
        )

        scan_transformed = o3d.geometry.PointCloud(scan)
        scan_transformed.transform(final_transform)
        aligned_clouds.append(scan_transformed)
        transforms[scan_path.name] = final_transform.tolist()

        export_path = args.export_dir / f"aligned_{scan_path.stem}.ply"
        o3d.io.write_point_cloud(str(export_path), scan_transformed)

    if not aligned_clouds:
        raise SystemExit("No scans were aligned successfully.")

    merged = aligned_clouds[0]
    for cloud in aligned_clouds[1:]:
        merged += cloud
    o3d.io.write_point_cloud(str(args.output), merged)
    print(f"Merged cloud written to {args.output}")

    with args.transforms.open("w", encoding="utf-8") as fh:
        json.dump(transforms, fh, indent=2)
    print(f"Transforms saved to {args.transforms}")


def main() -> None:
    align_scans(parse_args())


if __name__ == "__main__":
    main()
