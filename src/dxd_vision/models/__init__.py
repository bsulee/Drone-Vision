"""Data models for DXD Vision Engine."""

from dxd_vision.models.frame import FrameMetadata, FrameData, VideoInfo, ExtractionResult
from dxd_vision.models.detection import (
    BoundingBox,
    Detection,
    FrameDetections,
    DetectionSummary,
    ProcessingResult,
)

__all__ = [
    "FrameMetadata",
    "FrameData",
    "VideoInfo",
    "ExtractionResult",
    "BoundingBox",
    "Detection",
    "FrameDetections",
    "DetectionSummary",
    "ProcessingResult",
]
