"""
Configuration loader — reads YAML config and provides typed access.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    """Typed wrapper around the config dict with dot-access."""
    _data: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._data.get(key)
        if isinstance(val, dict):
            return Config(_data=val)
        if val is None:
            raise AttributeError(f"Config key '{key}' not found")
        return val

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls(_data=load_config(path))
