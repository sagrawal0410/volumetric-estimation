# ZED 2i Live Capture & Deployment

This guide covers the **ZED 2i deployment layer** in `volrecon`: live stereo RGB capture, canonical manifest export, FoundationStereo depth, and plain/weighted TSDF reconstruction.

## Hard rule: no ZED SDK depth for inference

This project **never** uses ZED SDK depth, point clouds, spatial mapping, or confidence maps as model input. Allowed ZED outputs:

- Rectified left/right RGB
- Camera intrinsics and stereo extrinsics / baseline
- Timestamps
- Optional pose metadata (positional tracking or external calibration)

Depth for reconstruction comes from **FoundationStereo** (or the repo stereo wrapper), not from `retrieve_measure`.

## 1. Install ZED SDK and Python API

1. Install [ZED SDK](https://www.stereolabs.com/developers/release/) for your OS/CUDA version.
2. Install the Python API (`pyzed.sl`) bundled with the SDK.
3. Install repo dependencies:

```bash
pip install -r requirements.txt
```

For CI or development without hardware:

```bash
export VOLRECON_MOCK_ZED=1
```

## 2. Confirm camera

```bash
python -m volrecon.scripts.zed_inspect
python -m volrecon.scripts.zed_inspect --dry-run   # calibration only, no PNG save
python -m volrecon.scripts.zed_inspect --mock       # mock camera for CI
```

Prints SDK version, serial, intrinsics, baseline, and saves one left/right pair to `debug/zed_inspect/`.

## 3. Capture only (RGB + manifest)

```bash
python -m volrecon.scripts.zed_capture_scene \
  --out data/zed_captures \
  --scene_name test_pile_001 \
  --resolution HD1080 \
  --fps 15 \
  --num_keyframes 30 \
  --pose_mode zed_tracking \
  --min_translation_between_keyframes_m 0.03 \
  --min_rotation_between_keyframes_deg 5 \
  --save_preview_video
```

Output layout:

```
data/zed_captures/<scene_name>/
  scene_meta.json
  camera_info.json
  calibration.json
  manifest.jsonl
  preview.mp4          # optional
  views/000000/{left,right}.png, meta.json
  runs/...
```

Config file: `configs/zed2i_capture.yaml`

## 4. End-to-end plain TSDF

```bash
python -m volrecon.scripts.zed_run_capture_then_reconstruct \
  --out data/zed_captures \
  --scene_name test_pile_001 \
  --resolution HD1080 \
  --fps 15 \
  --num_keyframes 30 \
  --pose_mode zed_tracking \
  --foundationstereo_repo /path/to/FoundationStereo \
  --foundationstereo_ckpt /path/to/model_best_bp2.pth \
  --method plain_tsdf \
  --voxel_length_m 0.003 \
  --sdf_trunc_m 0.015 \
  --depth_min_m 0.2 \
  --depth_max_m 4.0
```

Or use `configs/zed2i_live_plain_tsdf.yaml` with `--config`.

## 5. End-to-end weighted TSDF

```bash
python -m volrecon.scripts.zed_run_live_weighted_tsdf \
  --out data/zed_captures \
  --scene_name test_pile_001 \
  --resolution HD1080 \
  --fps 15 \
  --pose_mode zed_tracking \
  --foundationstereo_repo /path/to/FoundationStereo \
  --foundationstereo_ckpt /path/to/model_best_bp2.pth \
  --num_keyframes 40 \
  --voxel_length_m 0.003 \
  --sdf_trunc_m 0.015
```

Config: `configs/zed2i_live_weighted_tsdf.yaml`

## Why rectified left/right RGB?

FoundationStereo expects **rectified stereo pairs** with known intrinsics and baseline. The ZED SDK provides factory-calibrated, rectified LEFT/RIGHT views. Using raw unrectified images would require offline rectification and consistent calibration export.

## Why ZED depth is forbidden

- Keeps a single depth source (FoundationStereo) across ROBI, BOP, and live ZED pipelines.
- Avoids mixing SDK depth artifacts with learned stereo depth in TSDF fusion.
- Makes evaluation comparable: live capture has no GT; depth quality is entirely from the stereo model.

Runtime guards: `ZEDDepthCallGuard` wraps the camera; tests fail if forbidden API strings appear in `volrecon/camera`, `volrecon/deployment`, or `volrecon/scripts`.

## SVO for repeatable offline testing

Record (requires ZED SDK Recording API on device):

```bash
python -m volrecon.scripts.zed_record_svo \
  --out data/zed_svo/test_pile_001.svo2 \
  --resolution HD1080 \
  --fps 15 \
  --duration_sec 30
```

Extract RGB-only scene:

```bash
python -m volrecon.scripts.zed_extract_svo_rgb \
  --svo data/zed_svo/test_pile_001.svo2 \
  --out data/zed_captures/test_pile_001_from_svo \
  --frame_stride 5 \
  --pose_mode zed_tracking
```

## Fixed rig extrinsics (future truck-bay deployment)

Use `pose_mode=fixed_rig_yaml` with a `rig_calibration.yaml` mapping camera serial numbers to static `T_world_left`. For a single fixed camera, one static pose is saved per view. See `volrecon/camera/pose_sources.py`.

## Pose modes

| Mode | Use case |
|------|----------|
| `zed_tracking` | Hand-held camera orbiting an object/pile |
| `fixed_rig_yaml` | Fixed multi-ZED or single static camera rig |
| `external_poses` | COLMAP, AprilTag, robot FK, CSV/JSON poses |
| `none` | Capture without poses (single-view only with `--allow_no_pose_single_view`) |

## Resumability

All scripts support incremental runs:

- `--overwrite` — recapture images
- `--overwrite_depth` — rerun FoundationStereo
- `--overwrite_recon` — rerun TSDF

Existing outputs are skipped by default.

## Dry-run and mock modes

- `--dry-run` on inspect/capture: validate SDK + calibration without saving frames
- `--mock` or `VOLRECON_MOCK_ZED=1`: synthetic camera for tests/CI

## Limitations

- **No GT** for live capture unless you supply external scans/measurements
- **No truck wall/floor/background removal** yet — reconstructs everything in view
- **FoundationStereo** may fail on reflective, transparent, or very dark materials
- **Pose drift** from ZED tracking corrupts multi-view TSDF
- **Mesh volume** requires watertight mesh or falls back to voxel/hull estimates
- **Multi-view TSDF** needs camera motion poses or a calibrated fixed rig

## Tests

```bash
VOLRECON_MOCK_ZED=1 pytest tests/test_zed_*.py tests/test_live_pipeline_no_zed_depth_calls.py -v
```
