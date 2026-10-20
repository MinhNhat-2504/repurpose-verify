"""Configuration loading (TOML) with project-root-relative path resolution."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = field(default=PROJECT_ROOT)

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.raw[section]

    def get(self, section: str, default: Any = None) -> Any:
        return self.raw.get(section, default)

    def path(self, section: str, key: str) -> Path:
        p = Path(self.raw[section][key])
        return p if p.is_absolute() else self.root / p

    @property
    def cutoff(self) -> date:
        return date.fromisoformat(self.raw["time"]["cutoff"])

    @property
    def eval_end(self) -> date:
        return date.fromisoformat(self.raw["time"]["eval_end"])

    @property
    def cache_dir(self) -> Path:
        return self.path("paths", "cache_dir")

    @property
    def runs_dir(self) -> Path:
        return self.path("paths", "runs_dir")

    def model(self, name: str) -> dict[str, Any]:
        models = self.raw.get("models", {})
        if name not in models:
            raise KeyError(f"Unknown model profile '{name}'. Available: {sorted(models)}")
        return {"name": name, **models[name]}


def load_config(path: str | Path | None = None, overrides: dict | None = None,
                root: Path | None = None) -> Config:
    with open(DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, "rb") as f:
            raw = _deep_merge(raw, tomllib.load(f))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw=raw, root=root or PROJECT_ROOT)
