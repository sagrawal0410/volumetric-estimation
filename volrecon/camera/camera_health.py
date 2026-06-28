"""ZED camera health checks and depth-call guards."""

from __future__ import annotations

import inspect
from pathlib import Path

FORBIDDEN_ZED_PATTERNS = (
    "MEASURE.DEPTH",
    "MEASURE.XYZ",
    "retrieve_measure",
    "enable_spatial_mapping",
    "retrieve_spatial_map",
)

EXCLUDED_FROM_SCAN = frozenset(
    {
        "camera_health.py",
        "zed_mock.py",
    }
)

GUARDED_DIRS = ("volrecon/camera", "volrecon/deployment", "volrecon/scripts")


def assert_no_zed_depth_calls_enabled() -> None:
    """Runtime guard: ensure guarded modules do not contain forbidden ZED depth API strings."""
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for rel in GUARDED_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name in EXCLUDED_FROM_SCAN or "test_" in path.name:
                continue
            text = path.read_text(encoding="utf-8")
            for pat in FORBIDDEN_ZED_PATTERNS:
                if pat in text:
                    offenders.append(f"{path.relative_to(root)}: {pat}")
    if offenders:
        raise RuntimeError(
            "Forbidden ZED depth/spatial-mapping API references found:\n" + "\n".join(offenders)
        )


class ZEDDepthCallGuard:
    """Wrap a ZED camera object and raise if retrieve_measure is called."""

    def __init__(self, camera) -> None:
        self._camera = camera

    def __getattr__(self, name: str):
        if name == "retrieve_measure":
            raise RuntimeError(
                "retrieve_measure is forbidden in volrecon ZED capture. "
                "Use rectified LEFT/RIGHT RGB only; depth comes from FoundationStereo."
            )
        return getattr(self._camera, name)

    def __repr__(self) -> str:
        return f"ZEDDepthCallGuard({self._camera!r})"
