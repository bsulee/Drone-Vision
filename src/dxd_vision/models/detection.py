"""Detection data models for YOLO object detection."""

from pydantic import BaseModel
from typing import Optional


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x: int       # top-left x
    y: int       # top-left y
    width: int
    height: int


class Detection(BaseModel):
    """Single object detection."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    frame_number: int
    timestamp_ms: float


class FrameDetections(BaseModel):
    """All detections from a single frame."""
    frame_number: int
    timestamp_ms: float
    detections: list[Detection] = []

    @property
    def count(self) -> int:
        return len(self.detections)


class DetectionSummary(BaseModel):
    """Aggregated detection statistics."""
    total_detections: int
    by_class: dict[str, int]        # {"person": 23, "vehicle": 5}
    avg_confidence: float
    frames_with_detections: int
    frames_without_detections: int


class ProcessingResult(BaseModel):
    """Full pipeline result: extraction + optional detection."""
    video_info: "VideoInfo"
    frames_extracted: int
    extraction_fps: float
    sample_frame_path: Optional[str] = None
    detection_enabled: bool = False
    detection_summary: Optional[DetectionSummary] = None
    detections_path: Optional[str] = None       # JSON output path
    annotated_frame_path: Optional[str] = None   # sample with boxes


# Forward reference resolution
from dxd_vision.models.frame import VideoInfo
ProcessingResult.model_rebuild()
