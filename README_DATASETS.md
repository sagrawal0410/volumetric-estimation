# Datasets and Preprocessing

This document describes how **volrecon** preprocesses ROBI and BOP/T-LESS into a canonical format for multi-view volume reconstruction from **stereo RGB images**.

## Core constraint

During **inference**, the pipeline must **not** use provided depth maps as input. Depth may only be used for:

- evaluation
- validation
- debugging
- ground-truth mesh generation

The primary inference input is **2D RGB or stereo image pairs**. When a dataset does not provide true left/right stereo, the adapters report that honestly rather than fabricating pairs (except in the explicit synthetic sanity-check mode described below).

## Internal units

All geometry inside the processed format is stored in **meters**:

- BOP object models and translation vectors are converted from millimeters immediately after loading.
- Original units and raw metadata are preserved in per-view `meta.json` and manifest fields (`original_units`, `unit_conversion_applied`).

## Canonical layout

```
data/processed/
  manifests/
    robi_manifest.jsonl
    bop_tless_manifest.jsonl
  robi/<scene_id>/...
  bop_tless/<scene_id>/...
```

Each manifest row is one flattened `ViewRecord` with:

- image paths
- camera intrinsics `K`
- stereo calibration (if any)
- camera pose (if available)
- eval-only GT depth path
- object poses and model paths
- `inference_allowed_modalities` vs `eval_only_modalities`
- `synthetic` flag

## ROBI

ROBI layouts vary after download/unzip. The adapter:

1. Recursively scans the dataset root.
2. Classifies files by filename/folder tokens (`left`, `right`, `rgb`, `mono`, `depth`, etc.).
3. Writes `dataset_inspection_report_robi.md` documenting what was found.

### True stereo

If verifiable **left/right** pairs exist, views are marked `has_true_stereo=true` and are usable by stereo depth models such as FoundationStereo (subject to baseline/intrinsics metadata).

If only **monochrome stereo** pairs exist (`*_L`/`*_R` style), they are accepted under the `mono_stereo` modality.

If only **RGB + depth** exists, `has_true_stereo=false`. These views are suitable for multi-view RGB experiments only when poses exist, not for direct stereo-depth inference.

### Ground truth

- GT depth is stored under `gt_depth.png` and listed in **`eval_only_modalities` only**.
- Object models and scene meshes are copied/symlinked into `gt/object_models/` and `gt/scene_mesh_gt.ply` when present.
- If GT depth and poses exist but no scene mesh, a fusion stub is documented for later GT mesh generation.

## BOP / T-LESS

Standard BOP T-LESS provides **RGB/gray**, **depth**, **masks**, **camera intrinsics**, and **object poses**. It typically does **not** provide rectified left/right stereo pairs.

The adapter writes a strong warning in the inspection report and per-view notes:

> This split does not provide true stereo pairs; FoundationStereo inference cannot run directly without external stereo pairs or synthetic stereo rendering.

### Modes

#### `real_rgb_only` (default)

Extracts real RGB/gray frames, masks, intrinsics, eval-only GT depth, object poses, and models. **No fake left/right images** are created.

#### `synthetic_stereo_from_bop_mesh` (sanity check only)

For controlled algorithm/unit testing:

- Renders synthetic left/right pairs from GT meshes and poses.
- Default baseline: 0.06 m (configurable).
- All records are marked `synthetic=true`.
- **Do not mix these results with real-image benchmarks.**

### Multi-view grouping

- If `scene_camera.json` includes `cam_R_w2c` / `cam_t_w2c`, views are placed in a common world/scene frame.
- If world camera poses are missing, object-centric frames use GT object pose (`T_model_cam`) with model frame as canonical world for single-object instances.
- Cluttered multi-object scenes without world poses emit warnings that scene-level fusion is underdetermined.

### Rendered GT assets

For each BOP scene the extractor also builds:

- `gt/rendered_gt_depth/` — mesh-rendered depth per frame
- `gt/object_meshes_in_scene_frame/` — transformed meshes
- `gt/union_gt_voxels.npz` — union voxel grid at configurable resolution

## CLI

```bash
# Inspect raw layout
python -m volrecon.scripts.inspect_dataset --dataset robi --root /path/to/ROBI --out reports/robi_inspection.md

# Extract ROBI
python -m volrecon.scripts.extract_robi \
  --root /path/to/ROBI \
  --out data/processed/robi \
  --manifest data/processed/manifests/robi_manifest.jsonl \
  --symlink

# Extract BOP T-LESS (real images only)
python -m volrecon.scripts.extract_bop_tless \
  --root /path/to/tless \
  --split test_primesense \
  --mode real_rgb_only \
  --out data/processed/bop_tless \
  --manifest data/processed/manifests/bop_tless_manifest.jsonl \
  --symlink

# Synthetic stereo sanity-check mode
python -m volrecon.scripts.extract_bop_tless \
  --root /path/to/tless \
  --split test_primesense \
  --mode synthetic_stereo_from_bop_mesh \
  --baseline_m 0.06 \
  --out data/processed/bop_tless_synth_stereo \
  --manifest data/processed/manifests/bop_tless_synth_manifest.jsonl

# Validate processed manifest
python -m volrecon.scripts.validate_processed_dataset \
  --manifest data/processed/manifests/robi_manifest.jsonl
```

## FoundationStereo usability

A view is considered FoundationStereo-usable when:

- `stereo.has_true_stereo=true`
- both `left_path` and `right_path` exist
- intrinsics `K` are present
- for synthetic records, `baseline_m` is set and `synthetic=true`

Standard BOP T-LESS **`real_rgb_only`** views are **not** FoundationStereo-usable unless you supply external stereo pairs or use the synthetic sanity-check mode.

## Validation

`validate_processed_dataset.py` checks:

- all referenced paths exist
- image dimensions are consistent with `K`
- GT depth never appears in `inference_allowed_modalities`
- stereo pairs have matching dimensions
- BOP object model paths exist

Run tests:

```bash
pytest tests/
```
