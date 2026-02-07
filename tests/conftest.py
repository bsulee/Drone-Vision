"""Shared test fixtures for DXD Vision Engine."""

import cv2
import numpy as np
import pytest

from dxd_vision.config.settings import DXDConfig, ExtractionConfig


def _create_test_video(path, num_frames: int, fps: float = 30.0, width: int = 640, height: int = 480):
    """Helper to create a synthetic test video with frame numbers burned in."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame, f"Frame {i}", (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
        )
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a 10-second, 30fps, 640x480 test video with frame numbers burned in."""
    path = tmp_path / "test_video.mp4"
    return _create_test_video(path, num_frames=300, fps=30.0)


@pytest.fixture
def short_video_path(tmp_path):
    """Create a 2-second, 30fps, 640x480 test video."""
    path = tmp_path / "short_video.mp4"
    return _create_test_video(path, num_frames=60, fps=30.0)


@pytest.fixture
def default_config():
    """Return a DXDConfig with test-appropriate defaults."""
    return DXDConfig(extraction=ExtractionConfig(
        target_fps=5.0,
        output_dir="./test_output",
        save_sample_frame=True,
    ))


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)
