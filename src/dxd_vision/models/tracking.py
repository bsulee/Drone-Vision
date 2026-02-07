"""Tracking data models for multi-object tracking across frames."""

from typing import Optional

from pydantic import BaseModel

from dxd_vision.models.detection import BoundingBox


class TrackedDetection(BaseModel):
    """Single detection with a persistent track ID."""
    track_id: int
    object_id: str          # "{class}_{track_id}" e.g. "person_42"
    class_name: str
    confidence: float
    bbox: BoundingBox
    frame_number: int
    timestamp_ms: float


class FrameTracking(BaseModel):
    """All tracked detections from a single frame."""
    frame_number: int
    timestamp_ms: float
    tracked_detections: list[TrackedDetection] = []

    @property
    def count(self) -> int:
        return len(self.tracked_detections)


class ObjectTrajectory(BaseModel):
    """Full trajectory of a single tracked object across frames."""
    track_id: int
    object_id: str          # "{class}_{track_id}"
    class_name: str
    first_frame: int
    last_frame: int
    total_frames: int       # frames this object was visible
    avg_confidence: float
    positions: list[BoundingBox]     # bbox per frame (ordered)
    frame_numbers: list[int]         # corresponding frame numbers


class TrackingSummary(BaseModel):
    """Aggregated tracking statistics."""
    total_unique_objects: int
    by_class: dict[str, int]         # {"person": 5, "vehicle": 2}
    avg_track_length: float          # average frames per track
    longest_track: int               # max frames any single object was tracked
    total_detections: int            # sum of all per-frame detections
    frames_with_tracks: int
    frames_without_tracks: int


class TrackingResult(BaseModel):
    """Full pipeline result: extraction + tracking."""
    video_info: "VideoInfo"
    frames_extracted: int
    extraction_fps: float
    sample_frame_path: Optional[str] = None
    tracking_enabled: bool = True
    tracking_summary: Optional[TrackingSummary] = None
    tracking_path: Optional[str] = None         # JSON output path
    annotated_frame_path: Optional[str] = None  # sample with IDs + boxes


# Forward reference resolution
from dxd_vision.models.frame import VideoInfo  # noqa: E402
TrackingResult.model_rebuild()
