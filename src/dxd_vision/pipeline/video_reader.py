"""Video file reader with OpenCV."""

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from dxd_vision.config.settings import ExtractionConfig
from dxd_vision.models.frame import VideoInfo
from dxd_vision.pipeline.exceptions import (
    UnsupportedFormatError,
    VideoNotFoundError,
    VideoReadError,
)

logger = logging.getLogger(__name__)


class VideoReader:
    """OpenCV-based video file reader with validation and metadata extraction."""

    def __init__(self, video_path: str, config: ExtractionConfig):
        """Open video file, validate format, extract metadata.

        Args:
            video_path: Path to video file.
            config: Extraction configuration with supported formats.

        Raises:
            VideoNotFoundError: If file does not exist.
            UnsupportedFormatError: If file extension not in supported formats.
            VideoReadError: If OpenCV cannot open the file.
        """
        self._path = Path(video_path).resolve()
        self._config = config
        self._cap: cv2.VideoCapture | None = None

        # Validate file exists
        if not self._path.is_file():
            raise VideoNotFoundError(str(self._path))

        # Validate format
        suffix = self._path.suffix.lower()
        if suffix not in config.supported_formats:
            raise UnsupportedFormatError(str(self._path), config.supported_formats)

        # Open with OpenCV
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise VideoReadError(str(self._path), "OpenCV could not open file")

        logger.info("Opened video: %s", self._path.name)

    def get_video_info(self) -> VideoInfo:
        """Return VideoInfo with all metadata from the video file."""
        cap = self._ensure_open()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Decode fourcc to string
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

        return VideoInfo(
            path=str(self._path),
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration,
            codec=codec,
        )

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_number, frame_image) for every frame in the video.

        Frame numbers are 0-indexed. Resets to the beginning before reading.
        """
        cap = self._ensure_open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_number, frame
            frame_number += 1

        logger.info("Read %d frames from %s", frame_number, self._path.name)

    def release(self) -> None:
        """Release OpenCV VideoCapture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Released video capture: %s", self._path.name)

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def _ensure_open(self) -> cv2.VideoCapture:
        """Return the capture object, raising if already released."""
        if self._cap is None or not self._cap.isOpened():
            raise VideoReadError(str(self._path), "capture already released")
        return self._cap
