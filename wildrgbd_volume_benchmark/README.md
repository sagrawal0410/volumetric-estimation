# WildRGB-D Volume Benchmark

Benchmark object volume estimation on [WildRGB-D](https://github.com/wildrgbd/wildrgbd) using RGB-D videos, masks, real-scale poses, and **full-video pseudo-ground-truth**.

## Evaluation protocol

1. Fuse **all (or strided) frames** from a scene → pseudo-GT volume from full reconstruction.
2. Sample **4–5 diverse sparse views** from the same scene.
3. Run `convex_hull`, `tsdf`, `voxel_carving` on sparse views only.
4. Compare against pseudo-GT (never exact scalar GT).

All pseudo-GT is labeled:

```json
"gt_type": "full_video_reconstruction_pseudo_gt"
```

## Dataset layout

```
WildRGB-D/
  <category>/
    scenes/
      scenes_<scene_id>/
        rgb/<frame_id>.png
        depth/<frame_id>.png
        masks/<frame_id>.png
        metadata          # or metadata.json — contains K, width, height
        cam_poses.txt     # frame_id + 16 floats, T_cam_to_world (OpenCV)
    types.json          # scene_id -> single | multi | hand
```

- Depth: `depth_m = uint16_png / 1000.0`
- Start with **`single`** object scenes only.

## Install

```bash
cd /path/to/volumetric-estimation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Official download (large!)

Full dataset ≈ **3.37 TB** zipped, ≈ **4 TB** extracted. Downloads are **per category archive** (not per scene).

```bash
git clone https://github.com/wildrgbd/wildrgbd.git external/wildrgbd
cd external/wildrgbd
python download.py --cat <category_name>
python download.py --cat all   # entire dataset — needs huge disk
```

## Recommended: category subset workflow

```bash
export PYTHONPATH=.

python -m wildrgbd_volume_benchmark.download_subset \
  --repo_dir external/wildrgbd \
  --work_dir /large_tmp/wildrgbd_work \
  --subset_root data/wildrgbd_subset \
  --samples_per_category 3 \
  --scene_types single \
  --categories box,bottle,cup,bowl,apple,potato,shoe,backpack,stuffed_toy \
  --delete_full_category_after_subset true
```

## Prepare one scene

```bash
python -m wildrgbd_volume_benchmark.prepare_scene \
  --wildrgbd_root data/wildrgbd_subset \
  --category apple \
  --scene_id scenes_000123 \
  --num_views 5 \
  --scene_types single \
  --out_dir prepared/apple/scenes_000123 \
  --gt_frame_stride 1 \
  --sample_min_angle_deg 20
```

## Evaluate all three methods

```bash
python -m wildrgbd_volume_benchmark.run_eval \
  --prepared_scene_dir prepared/apple/scenes_000123 \
  --methods convex_hull tsdf voxel_carving \
  --voxel_length 0.003 \
  --sdf_trunc 0.015 \
  --voxel_size 0.004
```

## Batch benchmark

```bash
python -m wildrgbd_volume_benchmark.run_batch \
  --wildrgbd_root data/wildrgbd_subset \
  --samples_per_category 3 \
  --scene_types single \
  --num_views 5 \
  --out_root experiments/wildrgbd_subset_5view
```

Outputs: `aggregate_results.csv`, `category_summary.csv`, `method_summary.csv`, plots under `plots/`.

## Methods

| Method | Expected behavior |
|--------|---------------------|
| `convex_hull` | Robust sanity check; overestimates concave objects |
| `tsdf` | Best when masks/depth/poses are good; needs watertight mesh for volume |
| `voxel_carving` | Visual hull; overestimates occluded concavities |

## Pitfalls

- Pseudo-GT ≠ exact CAD volume.
- `multi` / `hand` scenes contaminate volume unless masks isolate one object — use `single` first.
- Validate poses via fused point cloud alignment.
- Full TSDF pseudo-GT may be non-watertight → falls back to voxel occupancy.
- Category archives are huge; use `download_subset` and delete full categories after extracting samples.

## Tests

```bash
PYTHONPATH=. pytest tests/test_wildrgbd_*.py -q
```

## Coexistence with T-LESS

This repo also includes **`tless_volume_benchmark`** for T-LESS BOP (exact mesh GT). Use:

- **T-LESS** → CAD mesh ground truth, BOP format
- **WildRGB-D** → full-video pseudo-GT, in-the-wild RGB-D videos
