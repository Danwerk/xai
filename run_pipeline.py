"""Reproduce the focused thesis experiment; notebooks and legacy runs are unnecessary."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.common import ROOT, read_config
from src.study import run_study


def main():
    parser = argparse.ArgumentParser(description="Realised-credit adaptive residual correction")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=("all", "panama", "estonia"), default="all")
    parser.add_argument(
        "--selection", type=Path, help="Frozen selection.json for transfer-only evaluation"
    )
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if args.development_only and args.dataset != "panama":
        parser.error("--development-only requires --dataset panama")
    if args.dataset == "estonia" and args.selection is None:
        parser.error("--dataset estonia requires --selection")
    if args.selection and args.dataset != "estonia":
        parser.error("--selection is for transfer-only evaluation")
    datasets = ("panama", "estonia") if args.dataset == "all" else (args.dataset,)
    output = args.output_dir or ROOT / "results/thesis_runs" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    completed = run_study(read_config(), output, datasets, args.selection, args.development_only)
    print(completed)


if __name__ == "__main__":
    main()
