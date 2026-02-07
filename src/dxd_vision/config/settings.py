"""Configuration system with Pydantic validation."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class ExtractionConfig(BaseModel):
    """Configuration for frame extraction."""

    target_fps: float = 5.0
    output_dir: str = "./output"
    save_sample_frame: bool = True
    supported_formats: list[str] = [".mp4", ".mov", ".avi", ".mkv"]


class DXDConfig(BaseModel):
    """Top-level configuration. Grows with each phase."""

    extraction: ExtractionConfig = ExtractionConfig()


def load_config(path: str = "config/default.yaml") -> DXDConfig:
    """Load config from YAML, falling back to defaults for missing fields."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return DXDConfig(**data)
    return DXDConfig()
