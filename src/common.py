"""Paths, serialization and deterministic defaults."""
import json
import logging
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def setup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for rel in ["data/raw", "data/processed", "data/synthetic", "models/saved",
                "outputs/metrics", "outputs/figures", "outputs/predictions",
                "outputs/patient_reports", "reports", "docs", "notebooks"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def dump_json(value, path):
    def convert(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False), encoding="utf-8")
