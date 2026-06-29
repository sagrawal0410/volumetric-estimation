"""RTAB-Map database export helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPORT_INSTRUCTIONS = (
    "Could not auto-export geometry from RTAB-Map database. "
    "Please export Dense Cloud or Mesh as PLY from RTAB-Map/Database Viewer "
    "and pass --input exported.ply."
)


@dataclass
class ExportResult:
    success: bool
    exported_path: Path | None
    message: str


def _find_rtabmap_tool(tools_path: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = tools_path / name
        if candidate.exists() and candidate.is_file():
            return candidate
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def try_export_from_db(
    db_path: Path,
    out_dir: Path,
    rtabmap_tools_path: Path | None = None,
) -> ExportResult:
    """Attempt to export dense cloud from RTAB-Map .db using CLI tools."""
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        return ExportResult(False, None, f"Database not found: {db_path}")

    tools = rtabmap_tools_path or Path("/usr/local/bin")
    export_tool = _find_rtabmap_tool(tools, ["rtabmap-export", "rtabmap_export"])

    if export_tool is None:
        return ExportResult(False, None, EXPORT_INSTRUCTIONS)

    out_ply = out_dir / "rtabmap_exported_cloud.ply"
    cmd = [str(export_tool), "--cloud", str(out_ply), str(db_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and out_ply.exists():
            return ExportResult(True, out_ply, "Exported dense cloud via rtabmap-export")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Try rtabmap-databaseViewer batch mode (rarely available headless)
    db_viewer = _find_rtabmap_tool(tools, ["rtabmap-databaseViewer"])
    if db_viewer is not None:
        return ExportResult(
            False,
            None,
            EXPORT_INSTRUCTIONS + f" Found {db_viewer} but headless export is not supported.",
        )

    return ExportResult(False, None, EXPORT_INSTRUCTIONS)
