"""Extract RGB from SVO to canonical scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from volrecon.camera.zed_svo import extract_svo_to_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rectified RGB from ZED SVO.")
    parser.add_argument("--svo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Scene output directory")
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--pose_mode", default="zed_tracking")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    scene = extract_svo_to_scene(args.svo, args.out, args.frame_stride, args.pose_mode)
    print(f"Extracted scene: {scene}")


if __name__ == "__main__":
    main()
