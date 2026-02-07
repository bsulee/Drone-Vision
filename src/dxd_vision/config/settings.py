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


class DetectionConfig(BaseModel):
    """Configuration for YOLO object detection."""

    enabled: bool = False
    model_path: str = "yolov8n.pt"   # default: YOLOv8 nano
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: list[str] = ["person", "vehicle", "weapon", "package"]
    device: str = "auto"             # "auto", "cpu", "cuda", "mps"
    save_annotated_frame: bool = True
    save_detections_json: bool = True


class DXDConfig(BaseModel):
    """Top-level configuration. Grows with each phase."""

    extraction: ExtractionConfig = ExtractionConfig()
    detection: DetectionConfig = DetectionConfig()


def load_config(path: str = "config/default.yaml") -> DXDConfig:
    """Load config from YAML, falling back to defaults for missing fields."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return DXDConfig(**data)
    return DXDConfig()
