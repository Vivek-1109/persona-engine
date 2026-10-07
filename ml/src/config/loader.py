"""
Persona Engine — Configuration Loader
Loads and merges YAML configurations with environment overrides (local vs colab).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _find_ml_root() -> Path:
    """Resolve the root ml/ directory dynamically."""
    current = Path(__file__).resolve()
    # Walk up until we find directory containing 'configs' or 'src'
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    # Fallback to current working directory or environment variable
    env_root = os.environ.get("PERSONA_ML_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    return Path.cwd()


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dictionary `update` into `base`."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config(dict):
    """
    Dictionary subclass providing dot-notation attribute access
    alongside standard dict operations.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to standard Python dict."""
        res: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                res[k] = v.to_dict()
            elif isinstance(v, list):
                res[k] = [item.to_dict() if isinstance(item, Config) else item for item in v]
            else:
                res[k] = v
        return res

    def get_path(self, key_path: str, default: Any = None) -> Any:
        """
        Safely fetch nested keys using dot notation (e.g. 'paths.raw_data').
        """
        parts = key_path.split(".")
        current = self
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


def load_config(
    config_name: Optional[str] = "base.yaml",
    environment: Optional[str] = None,
    config_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    resolve_paths: bool = True,
) -> Config:
    """
    Loads one or more configuration files, applies environment overrides,
    and returns a nested Config object.

    Args:
        config_name: Main config filename (e.g. 'base.yaml', 'training.yaml',
                     or a list of names/paths).
        environment: Target environment ('local' or 'colab').
                     If None, checks PERSONA_ENV env var or defaults to config value.
        config_dir: Path to directory containing configs.
        overrides: Dict of extra key-value overrides to apply.
        resolve_paths: If True, resolves relative path keys relative to ml root.

    Returns:
        Config object with dot notation and dictionary access.
    """
    ml_root = _find_ml_root()
    cfg_dir = Path(config_dir).resolve() if config_dir else ml_root / "configs"

    merged: Dict[str, Any] = {}

    # Always start by loading base.yaml if it exists
    base_file = cfg_dir / "base.yaml"
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, base_data)

    # Load requested config if different from base.yaml
    if config_name and config_name != "base.yaml":
        target = cfg_dir / config_name if not Path(config_name).is_file() else Path(config_name)
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                target_data = yaml.safe_load(f) or {}
                merged = _deep_merge(merged, target_data)

    # Determine environment
    env_detected = environment or os.environ.get("PERSONA_ENV") or merged.get("environment", "local")
    merged["environment"] = env_detected

    # Colab environment adjustments
    if env_detected.lower() == "colab":
        # Ensure paths match Colab runtime expectations if not overridden
        if "paths" in merged and not overrides:
            merged["paths"]["workspace"] = "/content/persona-engine/ml"

    # Apply manual overrides if given
    if overrides:
        merged = _deep_merge(merged, overrides)

    # Resolve paths to absolute paths if requested
    if resolve_paths and "paths" in merged:
        base_path = ml_root
        for key, val in list(merged["paths"].items()):
            if isinstance(val, str) and not os.path.isabs(val):
                merged["paths"][key] = str((base_path / val).resolve())

    # Attach root reference for convenience
    merged["ml_root"] = str(ml_root)

    return Config(merged)
