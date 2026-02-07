"""Shared test fixtures for DXD Vision Engine."""

import cv2
import numpy as np
import pytest

from dxd_vision.config.settings import DXDConfig, ExtractionConfig

# Preferred codecs in order — mp4v fails on macOS, avc1/MJPG work.
_CODEC_ATTEMPTS = [
    ("avc1", ".mp4"),
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
]


def _create_test_video(path, num_frames: int, fps: float = 30.0, width: int = 640, height: int = 480):
    """Helper to create a synthetic test video with frame numbers burned in.

    Tries multiple codecs for cross-platform compatibility. Falls back from
    avc1 → mp4v → MJPG/avi until one works.
    """
    path = str(path)

    for codec, ext in _CODEC_ATTEMPTS:
        # Rewrite extension to match the working codec's container
        actual_path = str(path).rsplit(".", 1)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(actual_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                frame, f"Frame {i}", (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
            )
            timestamp_ms = (i / fps) * 1000
            cv2.putText(
                frame, f"{timestamp_ms:.0f}ms", (50, height // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2,
            )
            writer.write(frame)
        writer.release()
        return actual_path

    raise RuntimeError(f"No working video codec found for writing test video to {path}")


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
