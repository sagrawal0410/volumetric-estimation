# Uncertainty-Weighted Multi-View TSDF Fusion

This pipeline replaces **plain unweighted TSDF** with **confidence-weighted fusion** designed for reflective/cluttered scenes (ROBI, future recycling loads).

## Important constraints

- **Inference input:** stereo RGB only → FoundationStereo depth
- **Never** uses dataset-provided depth as input
- GT depth/mesh/voxels are **evaluation-only**
- Confidence is **heuristic/self-supervised** unless explicitly calibrated

## Pipeline

```
Stereo RGB → FoundationStereo depth/disparity
          → per-pixel uncertainty maps
          → multi-view consistency
          → weighted TSDF (+ optional log-odds occupancy)
          → mesh + volume
          → compare vs plain TSDF + GT
```

## Confidence components

For each pixel, compute `confidence_total ∈ [0,1]` from:

| Component | Signal |
|-----------|--------|
| `valid` | Finite disparity > min_disp |
| `c_lr` | Left↔right disparity consistency |
| `c_photo` | Photometric warp error |
| `c_range` | Depth-dependent uncertainty ∝ z² |
| `c_angle` | Grazing-angle downweight via normals |
| `c_texture` | Local texture / ambiguity |
| `c_sat` | Over/under-exposure heuristic |
| `c_mv` | Multi-view depth agreement |
| `c_temp` | Temporal stability (1.0 if N/A) |

Combined (configurable exponents):

```
c_total = valid × c_lr^α_lr × c_photo^α_photo × ... × c_mv^α_mv
weight  = clip(w_min + w_scale × c_total, 0, w_max_per_obs)
```

## Weighted TSDF update

Custom **DenseChunkedWeightedTSDF** (not Open3D's unweighted integrator):

```
tsdf_new = (W_old × tsdf_old + w_i × tsdf_obs) / (W_old + w_i)
W_new    = min(W_old + w_i, W_max)
var_new  ≈ 1 / (W_new + ε)
```

Robust Huber/Tukey kernel further downweights outliers when `W_old > W_min`.

## Command sequence

### 1. Extract dataset (if not done)

See [README_DATASETS.md](README_DATASETS.md).

### 2. FoundationStereo depth

```bash
python -m volrecon.scripts.run_foundation_stereo \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --foundationstereo_repo /path/to/FoundationStereo \
  --ckpt /path/to/model_best_bp2.pth \
  --out data/runs/plain_tsdf/robi/depth_predictions
```

### 3. Plain TSDF baseline

```bash
python -m volrecon.scripts.run_plain_tsdf \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --out data/runs/plain_tsdf/robi/reconstructions \
  --config configs/plain_tsdf_robi.yaml
```

### 4. Uncertainty maps

```bash
python -m volrecon.scripts.compute_uncertainty_maps \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --out data/runs/weighted_tsdf/robi/uncertainty \
  --config configs/weighted_tsdf_robi.yaml \
  --run_right_to_left \
  --tau_lr_px 1.5 --tau_photo 0.08 --tau_mv_m 0.005 \
  --k_neighbor_views 5
```

### 5. Weighted TSDF

```bash
python -m volrecon.scripts.run_weighted_tsdf \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --uncertainty_dir data/runs/weighted_tsdf/robi/uncertainty \
  --out data/runs/weighted_tsdf/robi/reconstructions \
  --config configs/weighted_tsdf_robi.yaml \
  --use_occupancy
```

### 6. Compare plain vs weighted

```bash
python -m volrecon.scripts.compare_plain_vs_weighted \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --plain_dir data/runs/plain_tsdf/robi/reconstructions \
  --weighted_dir data/runs/weighted_tsdf/robi/reconstructions \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --uncertainty_dir data/runs/weighted_tsdf/robi/uncertainty \
  --out data/runs/weighted_tsdf/robi/comparison_report
```

### 7. Tune weights (validation only)

```bash
python -m volrecon.scripts.tune_uncertainty_weights \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --depth_predictions data/runs/plain_tsdf/robi/depth_predictions \
  --out data/runs/weighted_tsdf/robi/tuning
```

## Outputs per scene

**Uncertainty (`uncertainty/<scene>/<view>/`):**
- `confidence_total.npy`, `weight_total.npy`
- Component maps + `confidence_debug.png`

**Weighted reconstruction (`reconstructions/<scene>/`):**
- `mesh_weighted_raw.ply`, `mesh_weighted_clean.ply`
- `tsdf_grid.npz`, `weight_grid.npz`, `occupancy_grid.npz`
- `volume.json`, `report.html`

**Comparison (`comparison_report/`):**
- `aggregate_comparison.csv`
- Per-scene `metrics.json`, error-rejection curves

## BOP synthetic sanity test

Use manifest from `synthetic_stereo_from_bop_mesh` extraction + `configs/weighted_tsdf_bop_synth.yaml`.

Standard BOP `real_rgb_only` **cannot** run this pipeline without true/synthetic stereo.

## Limitations

- Confidence is heuristic unless calibrated on held-out data
- FoundationStereo may fail on extreme specular/transparent regions despite saturation downweighting
- Mesh volume unreliable when not watertight — use occupancy volume as alternate
- `SparseHashWeightedTSDF` is stubbed; use dense backend for bin-scale scenes
- Multi-view agreement requires camera poses; otherwise `c_mv = 1` with warning

## Tests

```bash
pytest tests/test_weighted_fusion.py -v
```
