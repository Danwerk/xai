"""Generate the one thesis report from a completed focused run; no training."""

import argparse
from pathlib import Path
from .common import latest_run
from .reporting import build_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    build_report(args.run or latest_run())
    print("Updated reports/ from the frozen run. No model was fitted.")
