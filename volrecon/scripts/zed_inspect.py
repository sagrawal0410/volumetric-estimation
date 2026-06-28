"""Inspect connected ZED camera and calibration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture
from volrecon.io.image_io import write_image

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ZED 2i camera (RGB only, no depth).")
    parser.add_argument("--resolution", default="HD720")
    parser.add_argument("--serial", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate SDK/calibration without saving images")
    parser.add_argument("--mock", action="store_true", help="Use mock ZED for CI")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    cfg = ZEDCaptureConfig(camera_resolution=args.resolution, serial_number=args.serial)
    capture = ZEDStereoCapture(cfg)
    capture.open()
    try:
        calib = capture.get_calibration()
        table = Table(title="ZED Calibration")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("SDK", capture._sdk_version)
        table.add_row("Serial", str(calib.get("camera_serial")))
        table.add_row("Baseline (m)", f"{calib['baseline_m']:.6f}")
        table.add_row("Resolution", f"{calib['image_width']}x{calib['image_height']}")
        table.add_row("fx left", f"{calib['K_left'][0,0]:.2f}")
        table.add_row("fy left", f"{calib['K_left'][1,1]:.2f}")
        console.print(table)

        if not args.dry_run:
            frame = capture.grab_frame()
            if frame:
                out = Path("debug/zed_inspect")
                out.mkdir(parents=True, exist_ok=True)
                write_image(out / "left.png", frame.left_rgb)
                write_image(out / "right.png", frame.right_rgb)
                console.print(f"Saved debug pair to {out.resolve()}")
    finally:
        capture.close()


if __name__ == "__main__":
    main()
