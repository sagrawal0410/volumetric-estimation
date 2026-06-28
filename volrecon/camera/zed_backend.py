"""ZED SDK backend with optional mock for CI."""

from __future__ import annotations

import os
from typing import Any

MOCK_ZED = os.environ.get("VOLRECON_MOCK_ZED", "").lower() in {"1", "true", "yes"}


def get_sl_module():
    if MOCK_ZED:
        from volrecon.camera import zed_mock as sl  # noqa: WPS433

        return sl
    try:
        import pyzed.sl as sl  # noqa: WPS433

        return sl
    except ImportError as exc:
        raise ImportError(
            "pyzed (ZED SDK Python API) is not installed. "
            "Install ZED SDK or set VOLRECON_MOCK_ZED=1 for mock mode."
        ) from exc
