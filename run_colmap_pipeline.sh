#!/usr/bin/env bash

# Run COLMAP's end-to-end photogrammetry pipeline on a folder of images.
# Usage:
#   ./run_colmap_pipeline.sh path/to/images output_workspace
#
# The script expects COLMAP to be on PATH. It will create/intermediate sparse
# and dense reconstructions, and produce a fused dense point cloud:
#   <workspace>/dense/fused.ply

set -euo pipefail

IMAGES_DIR=${1:-images}
WORKSPACE=${2:-colmap_workspace}
DB_PATH="$WORKSPACE/colmap.db"
SPARSE_DIR="$WORKSPACE/sparse"
DENSE_DIR="$WORKSPACE/dense"

if [ ! -d "$IMAGES_DIR" ]; then
  echo "Image directory '$IMAGES_DIR' does not exist" >&2
  exit 1
fi

mkdir -p "$WORKSPACE" "$SPARSE_DIR" "$DENSE_DIR"

echo "[1/6] Extracting features…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1

echo "[2/6] Matching features…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" exhaustive_matcher \
  --database_path "$DB_PATH"

echo "[3/6] Building sparse model…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR"

MODEL_PATH=$(ls "$SPARSE_DIR" | head -n1)
if [ -z "$MODEL_PATH" ]; then
  echo "COLMAP mapper did not produce a model" >&2
  exit 1
fi

echo "[4/6] Undistorting images…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" image_undistorter \
  --image_path "$IMAGES_DIR" \
  --input_path "$SPARSE_DIR/$MODEL_PATH" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP

echo "[5/6] Running dense stereo…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" patch_match_stereo \
  --workspace_path "$DENSE_DIR" \
  --PatchMatchStereo.geom_consistency true

echo "[6/6] Fusing depth maps…"
"/mnt/c/Users/yohan/Downloads/colmap-x64-windows-nocuda/bin/colmap.exe" stereo_fusion \
  --workspace_path "$DENSE_DIR" \
  --output_path "$DENSE_DIR/fused.ply"

echo "Dense point cloud saved to $DENSE_DIR/fused.ply"
