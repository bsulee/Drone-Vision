"""Frame extraction with FPS decimation."""

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from dxd_vision.config.settings import ExtractionConfig
from dxd_vision.models.frame import ExtractionResult, FrameData, FrameMetadata
from dxd_vision.pipeline.video_reader import VideoReader

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts frames from video at a target FPS using timestamp-based selection."""

    def __init__(self, reader: VideoReader, config: ExtractionConfig):
        self._reader = reader
        self._config = config
        self._video_info = reader.get_video_info()

    def extract(self) -> Iterator[FrameData]:
        """Yield FrameData at target_fps using timestamp-based selection.

        Uses timestamps rather than simple modulo to handle variable-rate
        videos and avoid drift. For each target interval, the first frame
        whose timestamp meets or exceeds the next target time is selected.
        """
        source_fps = self._video_info.fps
        target_fps = min(self._config.target_fps, source_fps)

        # Special case: extracting at source FPS means yield every frame
        if abs(target_fps - source_fps) < 0.01:
            for frame_number, image in self._reader.read_frames():
                timestamp_ms = (frame_number / source_fps) * 1000.0
                metadata = FrameMetadata(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    source_fps=source_fps,
                    extraction_fps=target_fps,
                    width=self._video_info.width,
                    height=self._video_info.height,
                    source_path=self._video_info.path,
                )
                yield FrameData(image=image, metadata=metadata)
            return

        target_interval_ms = 1000.0 / target_fps
        next_target_ms = 0.0

        for frame_number, image in self._reader.read_frames():
            timestamp_ms = (frame_number / source_fps) * 1000.0

            if timestamp_ms >= next_target_ms:
                metadata = FrameMetadata(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    source_fps=source_fps,
                    extraction_fps=target_fps,
                    width=self._video_info.width,
                    height=self._video_info.height,
                    source_path=self._video_info.path,
                )
                yield FrameData(image=image, metadata=metadata)
                next_target_ms += target_interval_ms

    def extract_all(self) -> ExtractionResult:
        """Full extraction: extract frames, optionally save sample, return result."""
        frames_extracted = 0
        sample_path = None

        for frame_data in self.extract():
            if frames_extracted == 0 and self._config.save_sample_frame:
                sample_path = self.save_sample_frame(
                    frame_data, self._config.output_dir
                )
            frames_extracted += 1

        target_fps = min(self._config.target_fps, self._video_info.fps)
        logger.info(
            "Extracted %d frames at %.1f fps from %s",
            frames_extracted,
            target_fps,
            self._video_info.path,
        )

        return ExtractionResult(
            video_info=self._video_info,
            frames_extracted=frames_extracted,
            extraction_fps=target_fps,
            sample_frame_path=sample_path,
        )

    def save_sample_frame(self, frame: FrameData, output_dir: str) -> str:
        """Save frame as PNG, return file path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        filename = f"sample_frame_{frame.metadata.frame_number}.png"
        path = out / filename
        cv2.imwrite(str(path), frame.image)

        logger.debug("Saved sample frame: %s", path)
        return str(path)
