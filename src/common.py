"""Small shared helpers for configuration and reproducible CSV artifacts."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, (pd.Timestamp, Path)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def read_artifact_csv(path, **kwargs):
    """Restore saved float64 values exactly, including histogram split boundaries.

    This reader is for generated artifacts. Original-source parsing retains its
    own documented loader so existing experiment inputs are not reinterpreted.
    """
    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kwargs)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config():
    return read_json(CONFIG_PATH)


def config_reference(cfg):
    """Hash parsed settings, so JSON formatting alone does not change a run."""
    content = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "path": "configs/experiment.json",
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def read_run_config(record):
    """Use the single configuration; reject results made with different settings."""
    cfg = read_config()
    if record.get("config_ref") != config_reference(cfg):
        raise ValueError("Run settings differ from configs/experiment.json; reproduce the run.")
    return cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_run():
    pointer = read_json(ROOT / "results/latest_thesis_run.json")
    folder = ROOT / pointer["path"]
    if sha256(folder / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Run manifest does not match its pointer.")
    if read_json(folder / "manifest.json")["status"] != "complete":
        raise ValueError("The selected run is incomplete.")
    return folder
