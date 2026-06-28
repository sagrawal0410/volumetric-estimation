# Plain TSDF Baseline Pipeline

End-to-end baseline for multi-view volume reconstruction:

**Stereo RGB only → FoundationStereo depth → plain unweighted Open3D TSDF → mesh → volume → evaluation**

Provided dataset depth is **never** used as inference input. GT depth/meshes/voxels are evaluation-only.

## Pipeline overview

```
manifest.jsonl
    ↓  (left/right + K + baseline + poses)
FoundationStereo  →  depth_m.npy per view
    ↓
Plain TSDF (Open3D ScalableTSDFVolume)  →  mesh_clean.ply
    ↓
Volume + metrics vs GT
```

## Coordinate conventions

| Symbol | Meaning |
|--------|---------|
| `T_world_cam` | Maps homogeneous camera-frame points to world frame: `p_world = T_world_cam @ p_cam` |
| `T_cam_world` | Inverse: `p_cam = T_cam_world @ p_world` |
| Open3D extrinsic | **World → camera** → pass `T_cam_world` (or `inverse(T_world_cam)`) |

### BOP object-centric mode

When BOP scenes lack `cam_R_w2c` / `cam_t_w2c`, fusion uses the **model frame as world**:

- `T_world_cam = T_cam_model`
- Open3D extrinsic = `T_model_cam`

This is documented and tested in `tests/test_plain_baseline.py`.

## FoundationStereo

### Requirements per view

- Rectified **left** and **right** images
- Intrinsics `K`
- Stereo **baseline** (meters)
- `has_true_stereo=true` in manifest

### Error when stereo unavailable

```
No true stereo pair available for this record. Use ROBI true stereo,
external stereo, or BOP synthetic_stereo_from_bop_mesh mode.
```

Standard BOP `real_rgb_only` splits will hit this error — by design.

### Outputs per view (`depth_predictions/<scene>/<view>/`)

- `disparity.npy`, `depth_m.npy`
- `valid_mask.png`, `depth_colormap.png`
- `pointcloud_est.ply`, `stereo_debug.json`
- `K_scaled.json` (when `--scale != 1`)

### Modes

- **Subprocess** (default): calls FoundationStereo `scripts/run_demo.py`
- **Python API**: if import succeeds; set `VOLRECON_USE_PYRENDER=1` only for mesh rendering, not FS

Rendering uses headless-safe depth fallback unless `VOLRECON_USE_PYRENDER=1`.

## Plain TSDF

Uses Open3D `ScalableTSDFVolume` with estimated FoundationStereo depth only.

Scene bounds (default, fair benchmarking):

1. Backproject estimated depths to world frame
2. Subsample + percentile filter (1st/99th)
3. Expand by margin (default 0.05 m)

`--use_gt_bounds_for_debug` may use GT mesh bounds for sanity checks only.

### Outputs per scene (`reconstructions/<scene>/`)

- `mesh_raw.ply`, `mesh_clean.ply`, `fused_pointcloud.ply`
- `tsdf_config.yaml`, `frame_list.json`, `bounds.json`

## Example commands

### ROBI (true stereo required)

```bash
# 1. Depth prediction
python -m volrecon.scripts.run_foundation_stereo \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --foundationstereo_repo /path/to/FoundationStereo \
  --ckpt /path/to/model_best_bp2.pth \
  --out data/runs/plain_tsdf/robi/depth_predictions \
  --max_views_per_scene 20 \
  --min_depth_m 0.1 --max_depth_m 2.0 \
  --scale 0.5 --valid_iters 16

# 2. TSDF fusion
python -m volrecon.scripts.run_plain_tsdf \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --out data/runs/plain_tsdf/robi/reconstructions \
  --config configs/plain_tsdf_robi.yaml

# 3. Evaluation
python -m volrecon.scripts.evaluate_reconstruction \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --recon_dir data/runs/plain_tsdf/robi/reconstructions \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --out data/runs/plain_tsdf/robi/eval_report \
  --num_sample_points 100000 \
  --thresholds_m 0.001 0.002 0.005 0.010

# End-to-end
python -m volrecon.scripts.run_plain_baseline_end_to_end \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --foundationstereo_repo /path/to/FoundationStereo \
  --ckpt /path/to/model_best_bp2.pth \
  --out data/runs/plain_tsdf/robi_e2e \
  --config configs/plain_tsdf_robi.yaml \
  --max_scenes 5 --max_views_per_scene 20
```

### BOP/T-LESS synthetic stereo sanity test

Extract with synthetic stereo first (see [README_DATASETS.md](README_DATASETS.md)), then:

```bash
python -m volrecon.scripts.run_foundation_stereo \
  --manifest data/processed/manifests/bop_tless_synth_manifest.jsonl \
  --foundationstereo_repo /path/to/FoundationStereo \
  --ckpt /path/to/model_best_bp2.pth \
  --out data/runs/plain_tsdf/bop_synth/depth_predictions \
  --scale 0.5

python -m volrecon.scripts.run_plain_tsdf \
  --manifest data/processed/manifests/bop_tless_synth_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/bop_synth/depth_predictions \
  --out data/runs/plain_tsdf/bop_synth/reconstructions \
  --config configs/plain_tsdf_bop_synth.yaml
```

**Warning:** Synthetic stereo results are sanity checks only — do not compare with real-image benchmarks.

### Visualization

```bash
python -m volrecon.scripts.visualize_scene \
  --mesh data/runs/plain_tsdf/robi/reconstructions/scene_01/mesh_clean.ply \
  --gt_mesh data/processed/robi/scene_01/gt/scene_mesh_gt.ply
```

## Evaluation metrics

### Reconstruction (mesh vs GT)

- Chamfer-L1, accuracy, completeness
- F-score @ 1/2/5/10 mm
- Outlier fractions

### Depth (per view, eval-only GT)

- AbsRel, RMSE, MAE, bad-pixel rates

### Volume

- Absolute/relative volume error vs GT mesh or BOP union voxels

## Tests

```bash
pytest tests/test_plain_baseline.py -v
```

Covers: synthetic cube TSDF volume, world-frame stability, no GT-depth cheating, no-stereo error, BOP pose convention.

## Config files

- `configs/plain_tsdf_robi.yaml` — ROBI defaults
- `configs/plain_tsdf_bop_synth.yaml` — BOP synthetic stereo sanity test
