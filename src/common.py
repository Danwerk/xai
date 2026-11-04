"""Shared artifact handling; CSV is the only tabular storage format."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def manifest_core_sha256(path):
    """Identity of original results, unaffected by the two appended XAI studies."""
    value = read_json(path)
    for key in ("attribution_refinement", "attribution_transfer"):
        value.pop(key, None)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_config(path=None):
    return json.loads(Path(path or ROOT / "configs/experiment.json").read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_run(pointer="latest_run.json"):
    """Leia valmis katse; kontrollsumma seob osuti õige tulemusega."""
    path = ROOT / "results" / pointer
    if not path.exists():
        raise FileNotFoundError(
            "Kohalikud katsetulemused puuduvad. Käivita uuring README järjekorras; tulemusi GitHubis kaasa ei anta."
        )
    link = read_json(path)
    folder = Path(link["path"])
    if sha256(folder / "manifest.json") != link["manifest_sha256"]:
        raise ValueError(f"Katse kontrollsumma ei vasta osutile: {pointer}")
    if read_json(folder / "manifest.json")["status"] != "complete":
        raise ValueError(f"Katse pole lõpetatud: {folder}")
    return folder
