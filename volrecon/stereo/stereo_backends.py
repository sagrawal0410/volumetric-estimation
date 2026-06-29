"""Stereo depth backend detection and checkpoint resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

StereoBackendName = Literal["foundation_stereo", "fast_foundation_stereo"]

FOUNDATION_STEREO_CKPT_NAMES = ("model_best_bp2.pth",)
FAST_FS_CKPT_NAMES = ("model_best_bp2_serialize.pth",)


def is_fast_foundation_stereo_repo(repo: Path) -> bool:
    """True if repo looks like NVlabs/Fast-FoundationStereo (not classic FoundationStereo)."""
    repo = repo.resolve()
    run_demo = repo / "scripts" / "run_demo.py"
    if not run_demo.exists():
        return False
    text = run_demo.read_text(encoding="utf-8")
    return "--model_dir" in text and "model_best_bp2_serialize" in text


def is_classic_foundation_stereo_repo(repo: Path) -> bool:
    repo = repo.resolve()
    run_demo = repo / "scripts" / "run_demo.py"
    if not run_demo.exists():
        return False
    text = run_demo.read_text(encoding="utf-8")
    return "--ckpt_dir" in text and "FoundationStereo" in text


def detect_stereo_backend(
    repo: Path,
    ckpt: Path,
    backend: str = "auto",
) -> StereoBackendName:
    if backend not in {"auto", "foundation_stereo", "fast_foundation_stereo"}:
        raise ValueError(f"Unknown stereo backend: {backend}")

    if backend != "auto":
        return backend  # type: ignore[return-value]

    ckpt_name = ckpt.name.lower()
    if ckpt_name.endswith("_serialize.pth") or "serialize" in ckpt_name:
        return "fast_foundation_stereo"
    if is_fast_foundation_stereo_repo(repo):
        return "fast_foundation_stereo"
    return "foundation_stereo"


def resolve_checkpoint_path(ckpt: Path, backend: StereoBackendName) -> Path:
    """Resolve checkpoint file; accepts a directory (e.g. weights/20-30-48/)."""
    ckpt = ckpt.expanduser()
    if ckpt.is_file():
        return ckpt.resolve()

    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")

    preferred = FAST_FS_CKPT_NAMES if backend == "fast_foundation_stereo" else FOUNDATION_STEREO_CKPT_NAMES
    for name in preferred:
        candidate = ckpt / name
        if candidate.exists():
            return candidate.resolve()

    pths = sorted(ckpt.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoint found under {ckpt}")
    return pths[0].resolve()


def require_cfg_yaml(ckpt_file: Path) -> Path:
    cfg = ckpt_file.parent / "cfg.yaml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Missing cfg.yaml next to checkpoint: expected {cfg}. "
            "Download the full weight folder from the model zoo, not only the .pth file."
        )
    return cfg
