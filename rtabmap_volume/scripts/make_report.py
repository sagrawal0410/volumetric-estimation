"""Regenerate HTML report from existing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtabmap_volume.viz.html_report import generate_html_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate HTML report from run directory")
    p.add_argument("--run_dir", required=True)
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    volume = json.loads((run / "reports" / "volume.json").read_text())
    log_path = run / "logs" / "processing_log.json"
    command = ""
    if log_path.exists():
        command = json.loads(log_path.read_text()).get("command", "")

    generate_html_report(
        run / "reports" / "report.html",
        {
            "input_path": str(run / "inputs" / "copied_input_geometry.ply"),
            "command": command,
            "config_path": str(run / "inputs" / "config_used.yaml"),
            "geometry_stats": {},
            "final_volume_m3": volume.get("final_volume_m3"),
            "final_volume_liters": volume.get("final_volume_liters"),
            "confidence": volume.get("confidence"),
            "confidence_score": volume.get("confidence_score_0_1"),
            "recommended_estimator": volume.get("recommended_estimator"),
            "upper_bound_m3": volume.get("upper_bound_m3"),
            "all_estimates": volume.get("all_estimates", {}),
            "warnings": volume.get("warnings", []),
        },
        run / "screenshots",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
