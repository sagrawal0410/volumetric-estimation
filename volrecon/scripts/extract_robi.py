"""Extract ROBI dataset to canonical processed format."""

from __future__ import annotations

import argparse
from pathlib import Path

from volrecon.config import DEFAULT_MANIFEST_DIR, DEFAULT_PROCESSED_ROOT, PreprocessConfig
from volrecon.datasets.robi import extract_robi


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ROBI to canonical processed format.")
    parser.add_argument("--root", required=True, type=Path, help="Path to raw ROBI dataset")
    parser.add_argument("--out", default=DEFAULT_PROCESSED_ROOT / "robi", type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_DIR / "robi_manifest.jsonl", type=Path)
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = PreprocessConfig(
        symlink=args.symlink or not args.copy,
        overwrite=args.overwrite,
        processed_root=args.out.parent,
    )

    extract_robi(
        root=args.root,
        out_dir=args.out,
        manifest_path=args.manifest,
        cfg=cfg,
    )
    print(f"ROBI extraction complete. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
