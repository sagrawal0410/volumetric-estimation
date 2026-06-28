"""Record ZED SVO."""

from __future__ import annotations

import argparse
import os

from volrecon.camera.zed_svo import SVORecordConfig, record_svo
from volrecon.camera.zed_capture import ZEDCaptureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Record ZED SVO (no depth sidecar).")
    parser.add_argument("--out", required=True, help="Output .svo2 path")
    parser.add_argument("--resolution", default="HD1080")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration_sec", type=float, default=30)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    record_svo(SVORecordConfig(__import__("pathlib").Path(args.out), args.resolution, args.fps, args.duration_sec))


if __name__ == "__main__":
    main()
