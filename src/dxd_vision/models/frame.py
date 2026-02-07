"""Core data models for video frame processing."""

from typing import Optional

import numpy as np
from pydantic import BaseModel


class FrameMetadata(BaseModel):
    """Metadata for a single extracted frame."""

    model_config = {"arbitrary_types_allowed": True}

    frame_number: int
    timestamp_ms: float
    source_fps: float
    extraction_fps: float
    width: int
    height: int
    source_path: str


class FrameData:
    """Container for frame image + metadata. Not Pydantic (holds numpy array)."""

    __slots__ = ("image", "metadata")

    def __init__(self, image: np.ndarray, metadata: FrameMetadata):
        self.image = image
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"FrameData(frame={self.metadata.frame_number}, "
            f"{self.metadata.width}x{self.metadata.height})"
        )


class VideoInfo(BaseModel):
    """Metadata about a source video file."""

    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    codec: str


class ExtractionResult(BaseModel):
    """Result of a frame extraction run."""

    video_info: VideoInfo
    frames_extracted: int
    extraction_fps: float
    sample_frame_path: Optional[str] = None
