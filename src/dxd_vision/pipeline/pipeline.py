"""Vision pipeline orchestrator."""

import logging

from dxd_vision.config.settings import DXDConfig
from dxd_vision.models.frame import ExtractionResult
from dxd_vision.pipeline.extractor import FrameExtractor
from dxd_vision.pipeline.video_reader import VideoReader

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrates video processing: VideoReader -> FrameExtractor -> results.

    Thin wrapper that wires components together. Grows through Phases 2-7.
    """

    def __init__(self, config: DXDConfig):
        self._config = config

    def process_video(self, video_path: str) -> ExtractionResult:
        """Open video, extract frames at target FPS, return results."""
        logger.info("Processing video: %s", video_path)

        with VideoReader(video_path, self._config.extraction) as reader:
            extractor = FrameExtractor(reader, self._config.extraction)
            result = extractor.extract_all()

        logger.info(
            "Pipeline complete: %d frames extracted", result.frames_extracted
        )
        return result
