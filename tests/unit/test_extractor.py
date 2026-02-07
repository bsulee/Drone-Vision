"""Unit tests for FrameExtractor."""

import os

import pytest

from dxd_vision.config.settings import DXDConfig, ExtractionConfig
from dxd_vision.models.frame import ExtractionResult, FrameData
from dxd_vision.pipeline.video_reader import VideoReader

try:
    from dxd_vision.pipeline.extractor import FrameExtractor
except ImportError:
    pytestmark = pytest.mark.skip(reason="FrameExtractor not yet implemented")


class TestFrameExtractor:
    """Tests for FrameExtractor — FPS decimation and frame metadata."""

    def test_extract_at_5fps_from_30fps(self, sample_video_path, default_config):
        """Extracting at 5fps from 30fps/10s video should yield ~50 frames."""
        config = ExtractionConfig(target_fps=5.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            frames = list(extractor.extract())
            assert abs(len(frames) - 50) <= 1

    def test_extract_at_1fps(self, sample_video_path):
        """Extracting at 1fps from 10s video should yield ~10 frames."""
        config = ExtractionConfig(target_fps=1.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            frames = list(extractor.extract())
            assert abs(len(frames) - 10) <= 1

    def test_extract_at_source_fps(self, sample_video_path):
        """Extracting at source FPS (30) should yield all 300 frames."""
        config = ExtractionConfig(target_fps=30.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            frames = list(extractor.extract())
            assert len(frames) == 300

    def test_frame_metadata_timestamps(self, sample_video_path, default_config):
        """Frame timestamps should be monotonically increasing."""
        config = ExtractionConfig(target_fps=5.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            timestamps = [f.metadata.timestamp_ms for f in extractor.extract()]
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i - 1], (
                    f"Timestamps not monotonic at index {i}: "
                    f"{timestamps[i - 1]} >= {timestamps[i]}"
                )

    def test_frame_metadata_dimensions(self, sample_video_path, default_config):
        """Frame metadata should report 640x480 dimensions."""
        config = ExtractionConfig(target_fps=5.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            for frame_data in extractor.extract():
                assert frame_data.metadata.width == 640
                assert frame_data.metadata.height == 480
                break  # Only need to check one

    def test_frame_metadata_source_path(self, sample_video_path, default_config):
        """Frame metadata should contain the source video path."""
        config = ExtractionConfig(target_fps=5.0)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            for frame_data in extractor.extract():
                assert frame_data.metadata.source_path
                break

    def test_save_sample_frame(self, sample_video_path, output_dir):
        """save_sample_frame should write a PNG file that exists on disk."""
        config = ExtractionConfig(target_fps=5.0, save_sample_frame=True)
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            first_frame = next(extractor.extract())
            path = extractor.save_sample_frame(first_frame, output_dir)
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_extract_all_returns_result(self, sample_video_path, output_dir):
        """extract_all should return an ExtractionResult."""
        config = ExtractionConfig(
            target_fps=5.0,
            save_sample_frame=True,
            output_dir=output_dir,
        )
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            result = extractor.extract_all()
            assert isinstance(result, ExtractionResult)

    def test_extract_all_frame_count_matches(self, sample_video_path, output_dir):
        """extract_all frame count should match manual extract() count."""
        config = ExtractionConfig(
            target_fps=5.0,
            save_sample_frame=False,
            output_dir=output_dir,
        )
        with VideoReader(sample_video_path, config) as reader:
            extractor = FrameExtractor(reader, config)
            result = extractor.extract_all()
            assert abs(result.frames_extracted - 50) <= 1
