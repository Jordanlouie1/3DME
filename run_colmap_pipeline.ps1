param(
    [string]$ImagesDir = "images",
    [string]$Workspace = "colmap_workspace",
    [string]$ColmapExe = "C:\Users\yohan\Downloads\colmap-x64-windows-nocuda\bin\colmap.exe"
)

if (-not (Test-Path $ImagesDir -PathType Container)) {
    throw "Image directory '$ImagesDir' does not exist."
}

if (-not (Test-Path $ColmapExe -PathType Leaf)) {
    throw "COLMAP executable not found at '$ColmapExe'."
}

$ColmapRoot = Split-Path -Parent (Split-Path -Parent $ColmapExe)
$PluginDir = Join-Path $ColmapRoot "plugins"
if (-not (Test-Path $PluginDir -PathType Container)) {
    throw "Qt plugin directory not found at '$PluginDir'."
}
$PreviousQtPluginPath = $env:QT_PLUGIN_PATH
$env:QT_PLUGIN_PATH = $PluginDir

$DBPath = Join-Path $Workspace "colmap.db"
$SparseDir = Join-Path $Workspace "sparse"
$DenseDir = Join-Path $Workspace "dense"

New-Item -ItemType Directory -Force -Path $Workspace, $SparseDir, $DenseDir | Out-Null

function Invoke-Colmap {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $ColmapExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP command failed: $($Arguments -join ' ')"
    }
}

Write-Host "[1/6] Extracting features..."
Invoke-Colmap -Arguments @(
    "feature_extractor",
    "--database_path", $DBPath,
    "--image_path", $ImagesDir,
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.single_camera", "0"
)

Write-Host "[2/6] Matching features..."
Invoke-Colmap -Arguments @(
    "exhaustive_matcher",
    "--database_path", $DBPath
)

Write-Host "[3/6] Building sparse model..."
Invoke-Colmap -Arguments @(
    "mapper",
    "--database_path", $DBPath,
    "--image_path", $ImagesDir,
    "--output_path", $SparseDir
)

$ModelPath = Get-ChildItem -Path $SparseDir -Directory | Select-Object -First 1
if (-not $ModelPath) {
    throw "COLMAP mapper did not produce a model in '$SparseDir'."
}

Write-Host "[4/6] Undistorting images..."
Invoke-Colmap -Arguments @(
    "image_undistorter",
    "--image_path", $ImagesDir,
    "--input_path", $ModelPath.FullName,
    "--output_path", $DenseDir,
    "--output_type", "COLMAP"
)

Write-Host "[5/6] Running dense stereo..."
Invoke-Colmap -Arguments @(
    "patch_match_stereo",
    "--workspace_path", $DenseDir,
    "--PatchMatchStereo.geom_consistency", "true"
)

Write-Host "[6/6] Fusing depth maps..."
$FusedPath = Join-Path $DenseDir "fused.ply"
Invoke-Colmap -Arguments @(
    "stereo_fusion",
    "--workspace_path", $DenseDir,
    "--output_path", $FusedPath
)

Write-Host "Dense point cloud saved to $FusedPath"

if ($null -ne $PreviousQtPluginPath) {
    $env:QT_PLUGIN_PATH = $PreviousQtPluginPath
} else {
    Remove-Item Env:QT_PLUGIN_PATH -ErrorAction SilentlyContinue
}
