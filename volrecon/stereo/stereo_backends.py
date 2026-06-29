"""Stereo depth backend detection and checkpoint resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

FAST_FS_REPO_URL = "https://github.com/NVlabs/Fast-FoundationStereo"
FOUNDATION_STEREO_REPO_URL = "https://github.com/NVlabs/FoundationStereo"

StereoBackendName = Literal["foundation_stereo", "fast_foundation_stereo"]

FOUNDATION_STEREO_CKPT_NAMES = ("model_best_bp2.pth",)
FAST_FS_CKPT_NAMES = ("model_best_bp2_serialize.pth",)


def resolve_repo_path(repo: Path) -> Path:
    """Expand ``~`` and normalize a stereo repo path."""
    return repo.expanduser().resolve()


def prepare_stereo_repo(repo: Path, *, backend: StereoBackendName | None = None) -> Path:
    """
    Validate a FoundationStereo / Fast-FoundationStereo clone and prepend it to ``sys.path``.

    Raises ``FileNotFoundError`` with clone instructions when the repo is missing or incomplete
    (e.g. only weights were downloaded without ``core/``).
    """
    repo = resolve_repo_path(repo)
    if not repo.is_dir():
        raise FileNotFoundError(
            f"Stereo repo not found: {repo}\n"
            f"Clone Fast-FoundationStereo: git clone {FAST_FS_REPO_URL} ~/Fast-FoundationStereo"
        )

    core_pkg = repo / "core" / "utils" / "utils.py"
    if not core_pkg.is_file():
        hint = FAST_FS_REPO_URL if backend != "foundation_stereo" else FOUNDATION_STEREO_REPO_URL
        raise FileNotFoundError(
            f"Incomplete stereo repo at {repo}: missing {core_pkg.relative_to(repo)}.\n"
            f"This usually means only weights were copied, not the full git clone.\n"
            f"Clone the repo: git clone {hint} {repo}"
        )

    if backend == "fast_foundation_stereo" and not (repo / "Utils.py").is_file():
        raise FileNotFoundError(
            f"Incomplete Fast-FoundationStereo repo at {repo}: missing Utils.py.\n"
            f"Clone the full repo: git clone {FAST_FS_REPO_URL} {repo}"
        )

    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def is_fast_foundation_stereo_repo(repo: Path) -> bool:
    """True if repo looks like NVlabs/Fast-FoundationStereo (not classic FoundationStereo)."""
    repo = resolve_repo_path(repo)
    run_demo = repo / "scripts" / "run_demo.py"
    if not run_demo.exists():
        return False
    text = run_demo.read_text(encoding="utf-8")
    return "--model_dir" in text and "model_best_bp2_serialize" in text


def is_classic_foundation_stereo_repo(repo: Path) -> bool:
    repo = resolve_repo_path(repo)
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
