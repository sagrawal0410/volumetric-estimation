"""Ensure sibling packages (tless_volume_benchmark, wildrgbd_volume_benchmark) are importable."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """
    Add the repository root to sys.path.

    Required when running ``python -m volume_benchmark.*`` without ``PYTHONPATH=.``
    so sibling packages like ``tless_volume_benchmark`` resolve.
    """
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
