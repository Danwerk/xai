"""Read the single experiment configuration; no result-file helpers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment.json"


def read_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
