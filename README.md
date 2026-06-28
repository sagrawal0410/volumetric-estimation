# volrecon

Multi-view volume reconstruction from stereo RGB images (ROBI + BOP/T-LESS).

See [README_DATASETS.md](README_DATASETS.md) for dataset preprocessing, [README_PLAIN_TSDF.md](README_PLAIN_TSDF.md) for the plain baseline, and [README_WEIGHTED_TSDF.md](README_WEIGHTED_TSDF.md) for uncertainty-weighted fusion.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

## Package layout

```
volrecon/
  config.py
  io/
  datasets/
  geometry/
  scripts/
tests/
```

Inference pipelines must not consume provided GT depth; depth is eval-only. See manifests `inference_allowed_modalities` vs `eval_only_modalities`.
