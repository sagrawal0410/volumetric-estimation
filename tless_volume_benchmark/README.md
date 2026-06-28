# T-LESS Volume Benchmark

Benchmark object volume estimation on the [T-LESS](https://bop.felk.cvut.cz/datasets/) dataset in BOP format.

Given 4–5 RGB-D views with visible object masks, per-frame intrinsics, and GT model-to-camera poses, this project prepares object-centric scans, runs three volume estimators, and compares predictions against volume computed from the official T-LESS 3D model.

## Methods

| Method | Description | Expected bias |
|--------|-------------|---------------|
| `convex_hull` | Fuse back-projected depth, convex hull volume | Overestimates non-convex objects |
| `tsdf` | NumPy TSDF fusion (+ optional Open3D) | Best when views, masks, depth, and poses are good |
| `voxel_carving` | Depth-aware visual hull carving | Overestimates occluded concavities |

## Download T-LESS (BOP format)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bop-benchmark/tless --local-dir data/bop/tless --repo-type=dataset

cd data/bop/tless
7z x tless_base.zip
7z x tless_models.zip -otless
7z x tless_train_primesense.zip -otless
# optional harder test split:
7z x tless_test_primesense_bop19.zip -otless
```

Expected layout after extraction:

```
data/bop/tless/tless/
  dataset_info.json
  models_cad/          # CAD models — use this for volume GT (default)
    models_info.json
    obj_000001.ply ... obj_000030.ply
  models_eval/         # decimated meshes for BOP pose-error metrics (same filenames)
  train_primesense/
    000001/
      rgb/ depth/ mask/ mask_visib/
      scene_camera.json scene_gt.json scene_gt_info.json
  test_primesense/   # or test_primesense_bop19/
    ...
```

- **`models_cad`** — manually created CAD models (BOP default for T-LESS). **Use for volume GT.**
- **`models_eval`** — same `obj_*.ply` names but uniformly decimated/resampled for ADD/ADI pose-error computation in BOP. Volumes are similar but not identical; avoid for ground-truth volume unless comparing to BOP eval meshes.
- **`models_reconst`** — optional; RGB-D reconstructed models with color (may appear in some downloads).

The benchmark auto-detects `models_cad` first. Override with `--model_dir models_cad` or `--model_preference eval`.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Prepare one object (train_primesense = clean isolated-object views)
python -m tless_volume_benchmark.tless_prepare \
  --dataset_root data/bop/tless/tless \
  --split train_primesense \
  --object_id 1 \
  --num_views 5 \
  --min_visib_fract 0.85 \
  --out_dir prepared/tless_obj_000001_train

# Evaluate all three methods
python -m tless_volume_benchmark.run_eval \
  --scan_dir prepared/tless_obj_000001_train \
  --methods convex_hull tsdf voxel_carving
```

## Batch benchmark

```bash
python -m tless_volume_benchmark.run_batch \
  --dataset_root data/bop/tless/tless \
  --split train_primesense \
  --object_ids 1,2,3,4,5,6,7,8,9,10 \
  --num_views 5 \
  --out_root experiments/tless_train_primesense
```

Outputs: `aggregate_summary.csv` and plots under `experiments/tless_train_primesense/plots/`.

## Prepared scan format

```
prepared/tless_obj_000001_train/
  gt_mesh.ply
  gt_volume.json
  selected_views.json
  frames/
    frame_000_rgb.png
    frame_000_depth.npy      # meters
    frame_000_mask.png
    frame_000_K.npy          # per-frame intrinsics
    frame_000_T_cam_to_object.npy
    frame_000_meta.json
  debug/
    frame_000_mask_overlay.png
    frame_000_depth_vis.png
    fused_points_by_view_colored.ply
  outputs/
    convex_hull/ tsdf/ voxel_carving/
    summary.csv
```

Object/model coordinates are the shared world frame.

## Units and BOP conventions

- **Depth PNG**: uint16; `depth_m = raw * depth_scale / 1000.0` (depth_scale from `scene_camera.json` converts raw → mm).
- **Model vertices & `cam_t_m2c`**: millimeters in BOP files → converted to meters internally.
- **`scene_gt.json`**: `cam_R_m2c`, `cam_t_m2c` map model → camera. Stored pose is `T_cam_to_object = inverse(T_model_to_camera)`.
- **Intrinsics**: read `cam_K` per image from `scene_camera.json` (do not assume a global K).
- **Masks**: instance masks named `{image_id:06d}_{gt_id:06d}.png`; prefer `mask_visib/` for visible surface.

## Train vs test splits

- **`train_primesense`**: Recommended first benchmark. One object per scene folder, high visibility, minimal clutter.
- **`test_primesense` / `test_primesense_bop19`**: Harder multi-object scenes with occlusion. Requires GT and visible masks in the downloaded archive.

## T-LESS pitfalls

- Textureless objects: RGB matching is hard, but depth + GT pose fusion works well.
- Always use **per-frame K** from `scene_camera.json`.
- Visible masks cover only the visible surface; full object volume requires multi-view fusion and comparison to the mesh GT.
- Non-watertight CAD models may use convex-hull fallback GT (labeled `exact_gt=false`).

## Troubleshooting

**Segmentation fault on `Running tsdf...`** — Open3D crashed. **Default TSDF backend is now pure NumPy** (no Open3D). Pull latest and run again; you should see `TSDF backend: numpy`.

Optional Open3D path: `export TLESS_TSDF_BACKEND=open3d` (only if your Open3D build is stable).

Install marching-cubes support:
```bash
pip install scikit-image
```

**View a PLY mesh:**
```bash
python -m tless_volume_benchmark.visualize_mesh prepared/tless_obj_000001_train/gt_mesh.ply
python -m tless_volume_benchmark.visualize_mesh prepared/tless_obj_000001_train/outputs/tsdf/tsdf_mesh_cleaned.ply
python -m tless_volume_benchmark.visualize_mesh prepared/tless_obj_000001_train/outputs/compare/tsdf_gt_vs_pred_side_by_side.ply
```

On a headless Linux server, use X forwarding (`ssh -X`) or copy PLYs to your laptop and open in [MeshLab](https://www.meshlab.net/) / Blender.

1. Diagnose:
```bash
python -m tless_volume_benchmark.doctor
```

2. Recreate the venv with native arm64 Python:
```bash
rm -rf .venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python -m tless_volume_benchmark.doctor
```

3. Run methods one at a time to see which crashes:
```bash
python -m tless_volume_benchmark.run_eval --scan_dir prepared/tless_obj_000001_train --methods convex_hull
python -m tless_volume_benchmark.run_eval --scan_dir prepared/tless_obj_000001_train --methods tsdf
python -m tless_volume_benchmark.run_eval --scan_dir prepared/tless_obj_000001_train --methods voxel_carving
```

4. Optional Open3D TSDF (only if numpy backend is not enough):
```bash
export TLESS_TSDF_BACKEND=open3d
python -m tless_volume_benchmark.run_eval --scan_dir ... --methods tsdf
```

`run_eval` prints progress per method. **`Running convex_hull...` + Killed** = OOM — use latest code or `--voxel_downsample 0.003`.

## Tests

```bash
PYTHONPATH=. pytest tests/test_bop_units.py tests/test_pose_conversion.py \
  tests/test_mesh_volume.py tests/test_methods_on_synthetic_prepared_scan.py -q
```

## Project layout

```
tless_volume_benchmark/
  io_bop.py           # BOP/T-LESS I/O and candidate iteration
  geometry.py         # Poses, backprojection, projection
  mesh_volume.py      # GT mesh loading and volume
  view_selection.py   # Diverse view selection
  tless_prepare.py    # Prepare CLI
  run_eval.py         # Single-scan evaluation
  run_batch.py        # Batch prepare + eval
  visualize.py        # Debug overlays
  visualize_mesh.py   # Interactive PLY viewer
  compare_results.py  # GT vs pred table + side-by-side PLYs
  methods/            # convex_hull, tsdf (numpy), tsdf_open3d, voxel_carving
```
