# volume_benchmark

Python 3.10+ benchmark for evaluating **object volume estimation** from 4–5 multi-view RGB-D / depth-camera observations.

All geometry uses **meters**. Depth maps are `float32` in meters. Poses are `T_cam_to_object` (4×4, OpenCV camera convention: x right, y down, z forward).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # optional: if using pyproject/setup; otherwise PYTHONPATH=.
export PYTHONPATH=.
```

## Normalized prepared scan format

```
scan_dir/
  K.npy
  gt_mesh.ply
  gt_volume.json
  metadata.json          # optional
  frames/
    frame_000_depth.npy
    frame_000_mask.png
    frame_000_T_cam_to_object.npy
    ...
```

| File | Description |
|------|-------------|
| `K.npy` | 3×3 pinhole intrinsics |
| `gt_mesh.ply` | Ground-truth or reference mesh |
| `gt_volume.json` | Volume metadata (`volume_m3`, `gt_type`, `watertight`, …) |
| `frame_*_depth.npy` | `float32` depth in **meters** |
| `frame_*_mask.png` | `uint8` PNG, 255 = object, 0 = background |
| `frame_*_T_cam_to_object.npy` | 4×4 pose mapping camera → object frame |

Ground-truth volume is only labeled `mesh_watertight` when the mesh is watertight (`abs(mesh.volume)`). Non-watertight meshes require `--repair-mesh` (→ `mesh_repaired`) or an explicit pseudo-GT method.

## CLI

### Prepare a dataset

From a YAML/JSON manifest:

```bash
python -m volume_benchmark.prepare_dataset from-manifest manifest.yaml --validate
```

Example manifest:

```yaml
dataset: ycb
output_dir: data/prepared/mustard_scan
mesh_path: models/006_mustard_bottle/textured.obj
mesh_units: m
repair_mesh: false
depth_scale: 0.001
K: [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]]
frames:
  - [raw/depth_0.png, raw/mask_0.png, raw/pose_0.txt]
  - [raw/depth_1.png, raw/mask_1.png, raw/pose_1.txt]
metadata:
  object_id: 006_mustard_bottle
```

Compute GT volume for a mesh only:

```bash
python -m volume_benchmark.prepare_dataset mesh-gt model.ply out/gt_only --mesh-units auto
```

Supported adapters: `bop`, `ycb` (see `volume_benchmark/datasets/`).

For T-LESS BOP volume benchmarking, see [`tless_volume_benchmark/README.md`](tless_volume_benchmark/README.md).

For WildRGB-D in-the-wild volume benchmarking, see [`wildrgbd_volume_benchmark/README.md`](wildrgbd_volume_benchmark/README.md).

### Evaluate one scan

```bash
python -m volume_benchmark.run_eval data/prepared/box_scan \
  --methods convex_hull tsdf voxel_carving \
  --num-views 5 \
  --output reports/box_scan.json
```

### Batch evaluation

```bash
python -m volume_benchmark.run_batch_eval data/prepared \
  --output-dir reports/batch \
  --methods convex_hull voxel_carving \
  --num-views 4
```

Writes `batch_results.csv`, `batch_results.json`, and `summary_by_method.csv`.

## Methods

| Method | Module | Description |
|--------|--------|-------------|
| `convex_hull` | `methods/convex_hull.py` | Fuse back-projected depth points; convex hull volume |
| `tsdf` | `methods/tsdf.py` | Open3D scalable TSDF fusion |
| `voxel_carving` | `methods/voxel_carving.py` | Visual-hull voxel carving |

## Tests

```bash
pytest tests/ -v
# skip slow Open3D TSDF test:
pytest tests/ -v -m "not slow"
```

## Project layout

```
volume_benchmark/
  common/          # io, geometry, mesh_volume, metrics, view_selection, visualization
  datasets/        # bop, ycb adapters
  tless_volume_benchmark/  # T-LESS BOP volume benchmark
  wildrgbd_volume_benchmark/  # WildRGB-D pseudo-GT volume benchmark
  methods/         # convex_hull, tsdf, voxel_carving
  prepare_dataset.py
  run_eval.py
  run_batch_eval.py
tests/
```

## Units cheat sheet

- Depth: **meters** (`float32`)
- Mesh vertices: **meters** after `load_mesh_as_meters`
- Volume: **m³** internally; reports also include **cm³**
- BOP/YCB/BigBird raw depth PNGs: typically **mm** → use `depth_scale=0.001`
