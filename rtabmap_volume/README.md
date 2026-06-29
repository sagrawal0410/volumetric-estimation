# RTAB-Map Volume Estimation Pipeline

Production-quality Python post-processing for **RTAB-Map** reconstructed geometry. Takes exported dense clouds or meshes and estimates object/pile volume with multiple fallback methods, honest confidence scoring, and HTML reports.

## Recommended workflow

1. Reconstruct your scene in **RTAB-Map** from rectified stereo RGB.
2. Export **Dense Cloud** or **Mesh** as `.ply` (File → Export, or Database Viewer).
3. Inspect the export:

```bash
python -m rtabmap_volume.scripts.inspect_geometry \
  --input exported.ply \
  --units m \
  --out runs/inspect_scene
```

4. If the scene includes table/floor/background, choose segmentation (crop, plane removal, or interactive crop).
5. Run volume estimation:

```bash
python -m rtabmap_volume.scripts.estimate_volume \
  --input exported.ply \
  --config rtabmap_volume/configs/high_accuracy.yaml \
  --out runs/volume/scene_001 \
  --units m \
  --segmentation plane_then_cluster \
  --overwrite
```

6. Open `runs/volume/scene_001/reports/report.html`.

## Installation

```bash
cd rtabmap_volume
pip install -r requirements.txt
pip install -e .
```

Requires **Python 3.10+**. Optional: `pymeshlab` for stronger mesh repair, `plotly` for interactive HTML charts.

## Primary CLI

```bash
python -m rtabmap_volume.scripts.estimate_volume \
  --input /path/to/rtabmap_exported_mesh_or_cloud.ply \
  --out runs/volume/test_scene_001 \
  --config rtabmap_volume/configs/recycling_pile.yaml \
  --units m \
  --segmentation height_above_plane \
  --roi_json optional_roi.json \
  --known_scale_object_json optional_scale.json \
  --overwrite
```

### RTAB-Map database input

```bash
python -m rtabmap_volume.scripts.estimate_volume \
  --rtabmap_db map.db \
  --rtabmap_tools_path /usr/local/bin \
  --out runs/from_db \
  --config rtabmap_volume/configs/object_on_table.yaml
```

If auto-export fails, the tool prints instructions to export PLY manually and re-run with `--input`.

## Output layout

```
runs/volume/test_scene_001/
├── inputs/
│   ├── copied_input_geometry.ply
│   └── config_used.yaml
├── processed/
│   ├── cloud_raw.ply
│   ├── cloud_denoised.ply
│   ├── cloud_cropped.ply
│   ├── cloud_object_segmented.ply
│   ├── mesh_repaired.ply
│   └── voxel_grid.npz
├── reports/
│   ├── volume.json
│   ├── volume.csv
│   └── report.html
├── screenshots/
└── logs/
    ├── warnings.txt
    └── processing_log.json
```

## Segmentation modes

| Mode | Use case |
|------|----------|
| `none` | Pre-isolated object mesh |
| `manual_aabb` / `manual_obb` | ROI from `--roi_json` |
| `interactive_crop` | Open3D visualizer → saves ROI JSON |
| `plane_then_cluster` | Object on table/floor (RANSAC + DBSCAN) |
| `height_above_plane` | Recycling piles, truck beds |

## Estimator meanings

| Method | Interpretation |
|--------|----------------|
| **direct_mesh_volume** | Reliable **only if watertight** |
| **repaired_mesh_volume** | Good when repair is modest; may hallucinate missing surfaces |
| **poisson_mesh_volume** | Model-based; can over-smooth noisy clouds |
| **ball_pivoting_mesh_volume** | Dense sampling; often non-watertight |
| **alpha_shape_volume** | Tunable hull; unstable if alpha-sensitive |
| **voxel_occupancy_volume** | Robust fallback; resolution-dependent |
| **heightfield_volume** | Best for **bulk/pile envelope** above support plane |
| **convex_hull_volume** | **Upper bound** — overestimates concave objects |
| **final consensus** | Selected by geometry quality + config priority |

## Config presets

- `configs/default.yaml` — balanced defaults
- `configs/object_on_table.yaml` — plane + cluster, mesh-first consensus
- `configs/recycling_pile.yaml` — height-field + voxel for bulk piles
- `configs/high_accuracy.yaml` — full methods, fine voxels, all intermediates
- `configs/fast_preview.yaml` — downsampled quick preview

## Other scripts

```bash
# Interactive crop
python -m rtabmap_volume.scripts.crop_interactive --input scene.ply --out roi.json

# Batch processing
python -m rtabmap_volume.scripts.batch_estimate_volume \
  --input_dir rtabmap_exports/ --out runs/batch --config rtabmap_volume/configs/recycling_pile.yaml

# Ground-truth evaluation
python -m rtabmap_volume.scripts.evaluate_against_gt --csv eval.csv --out runs/eval

# Synthetic smoke test
python -m rtabmap_volume.scripts.synthetic_volume_smoke_test
```

## Tests

```bash
pytest rtabmap_volume/tests -v
```

## Important warnings

- **RTAB-Map reconstruction quality** bounds final volume accuracy.
- **Bad scale/calibration** ruins volume — always verify `--units` and use scale references when possible.
- **Partial scans** cannot recover hidden sides without assumptions.
- Reflective, transparent, or black objects may be under-reconstructed.
- **Segmentation/cropping** is often the largest error source.
- For recycling/truck scenes, interpret results as **visible bulk envelope volume**, not true hidden material volume unless independently calibrated.

## License

Part of the volumetric-estimation repository.
