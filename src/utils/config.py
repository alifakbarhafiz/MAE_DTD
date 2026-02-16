"""Load YAML config with _base_ inheritance."""
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_path = path.parent / base
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
