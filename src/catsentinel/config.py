"""Loads config.yaml into a nested, attribute-accessible structure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin attribute-access wrapper around a nested dict loaded from YAML."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)
