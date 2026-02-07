"""Unit tests for VideoReader."""

import pytest

from dxd_vision.config.settings import ExtractionConfig
from dxd_vision.models.frame import VideoInfo
from dxd_vision.pipeline.exceptions import (
    UnsupportedFormatError,
    VideoNotFoundError,
    VideoReadError,
)
from dxd_vision.pipeline.video_reader import VideoReader


class TestVideoReader:
    """Tests for VideoReader — the OpenCV-based video file reader."""

    def test_opens_valid_mp4(self, sample_video_path, default_config):
        """VideoReader should open a valid video file without raising."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            assert reader is not None

    def test_returns_correct_video_info(self, sample_video_path, default_config):
        """get_video_info should return a VideoInfo instance."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            info = reader.get_video_info()
            assert isinstance(info, VideoInfo)

    def test_video_info_frame_count(self, sample_video_path, default_config):
        """10s at 30fps should yield 300 frames."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            info = reader.get_video_info()
            assert info.total_frames == 300

    def test_video_info_fps(self, sample_video_path, default_config):
        """Source FPS should be ~30.0."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            info = reader.get_video_info()
            assert info.fps == pytest.approx(30.0, abs=0.1)

    def test_video_info_dimensions(self, sample_video_path, default_config):
        """Dimensions should be 640x480."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            info = reader.get_video_info()
            assert info.width == 640
            assert info.height == 480

    def test_video_info_duration(self, sample_video_path, default_config):
        """Duration should be approximately 10.0 seconds."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            info = reader.get_video_info()
            assert info.duration_seconds == pytest.approx(10.0, abs=0.1)

    def test_read_frames_yields_all(self, sample_video_path, default_config):
        """read_frames should yield exactly 300 frames for a 10s/30fps video."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            frames = list(reader.read_frames())
            assert len(frames) == 300

    def test_read_frames_correct_shape(self, sample_video_path, default_config):
        """Each frame should be (480, 640, 3) — height x width x channels."""
        with VideoReader(sample_video_path, default_config.extraction) as reader:
            for _, frame in reader.read_frames():
                assert frame.shape == (480, 640, 3)
                break  # Only need to check one frame

    def test_context_manager_releases(self, sample_video_path, default_config):
        """After exiting context manager, capture should be released."""
        reader = VideoReader(sample_video_path, default_config.extraction)
        reader.__enter__()
        reader.__exit__(None, None, None)
        # After release, _cap should be None
        assert reader._cap is None

    def test_file_not_found_raises(self, default_config):
        """Opening a non-existent file should raise VideoNotFoundError."""
        with pytest.raises(VideoNotFoundError):
            VideoReader("/nonexistent/video.mp4", default_config.extraction)

    def test_unsupported_format_raises(self, tmp_path, default_config):
        """Opening a file with unsupported extension should raise UnsupportedFormatError."""
        bad_file = tmp_path / "video.xyz"
        bad_file.write_text("not a video")
        with pytest.raises(UnsupportedFormatError):
            VideoReader(str(bad_file), default_config.extraction)

    def test_corrupt_video_raises(self, tmp_path, default_config):
        """Opening a corrupt file with a valid extension should raise VideoReadError."""
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"\x00" * 256)
        with pytest.raises(VideoReadError):
            VideoReader(str(corrupt), default_config.extraction)
