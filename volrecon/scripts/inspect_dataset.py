"""Inspect raw dataset layout and write a report."""

from __future__ import annotations

import argparse
from pathlib import Path

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.bop import inspect_bop
from volrecon.datasets.robi import inspect_robi


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw ROBI or BOP/T-LESS dataset layout.")
    parser.add_argument("--dataset", required=True, choices=["robi", "bop_tless"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test_primesense", help="BOP split folder name")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "robi":
        inspect_robi(args.root.resolve(), args.out.resolve())
    else:
        inspect_bop(args.root.resolve(), args.split, args.out.resolve())
    print(f"Wrote inspection report to {args.out}")


if __name__ == "__main__":
    main()
