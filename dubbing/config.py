"""YAML-based experiment configuration utilities.

Usage
-----
Load a YAML file and get a nested attribute-accessible config::

    cfg = load_config("dubbing/configs/default.yaml")
    print(cfg.training.learning_rate)   # 0.0001
    print(cfg.model.DiT.hidden_dim)     # 512

Apply dotted CLI overrides::

    apply_overrides(cfg, ["training.learning_rate=5e-4", "system.gpu=1"])

Serialize back to dict (for JSON/YAML saving)::

    d = config_to_dict(cfg)
"""
from __future__ import annotations

import yaml
from types import SimpleNamespace
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to nested SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def _ns_to_dict(obj) -> object:
    """Recursively convert nested SimpleNamespace to plain dicts."""
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    return obj


def _parse_val(s: str):
    """Attempt to parse a CLI override string as bool / int / float / str."""
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.lower() in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> SimpleNamespace:
    """Load a YAML config file and return as a nested SimpleNamespace."""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return _dict_to_ns(d or {})


def apply_overrides(cfg: SimpleNamespace, overrides: list) -> SimpleNamespace:
    """Apply dotted ``key.subkey=value`` overrides to a nested namespace in-place.

    Example::

        apply_overrides(cfg, ["training.learning_rate=5e-4", "system.gpu=1"])
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override!r}")
        key, val = override.split("=", 1)
        keys = key.split(".")
        obj = cfg
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, SimpleNamespace())
            obj = getattr(obj, k)
        setattr(obj, keys[-1], _parse_val(val))
    return cfg


def config_to_dict(cfg) -> dict:
    """Serialize a nested SimpleNamespace to a plain dict (for JSON/YAML saving)."""
    return _ns_to_dict(cfg)
