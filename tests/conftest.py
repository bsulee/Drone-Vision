"""Shared test fixtures for DXD Vision Engine."""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from dxd_vision.config.settings import DXDConfig, DetectionConfig, ExtractionConfig
from dxd_vision.models.frame import FrameData, FrameMetadata

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


# ---------------------------------------------------------------------------
# Phase 2: Detection fixtures
# ---------------------------------------------------------------------------

class _MockBox:
    """Mimics a single ultralytics box with xyxy, conf, cls attributes."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.cls = np.array([cls], dtype=np.float32)


class _MockBoxes:
    """Mimics ultralytics Results.boxes for predictable test detections.

    Returns 2 'person' detections and 1 'car' detection.
    Iterable — yields individual _MockBox objects like real ultralytics.
    Iteration is dynamic: overriding xyxy/conf/cls arrays updates what
    __iter__ yields, so tests can create empty-detection scenarios.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.xyxy = np.array([
            [50, 80, 200, 400],    # person 1
            [300, 60, 450, 380],   # person 2
            [400, 200, 600, 350],  # car
        ], dtype=np.float32)
        self.conf = np.array([0.92, 0.85, 0.78], dtype=np.float32)
        self.cls = np.array([0, 0, 2], dtype=np.float32)  # COCO: 0=person, 2=car

    def __iter__(self):
        for i in range(len(self.xyxy)):
            yield _MockBox(self.xyxy[i], self.conf[i], self.cls[i])

    def __len__(self):
        return len(self.xyxy)


class _MockResult:
    """Mimics a single ultralytics Results object."""

    def __init__(self, width: int = 640, height: int = 480):
        self.boxes = _MockBoxes(width, height)


def _make_mock_yolo_model():
    """Build a mock that behaves like ultralytics.YOLO for testing."""
    model = MagicMock()

    # model.names: COCO class name mapping
    model.names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 24: "backpack", 26: "handbag",
        28: "suitcase", 43: "knife",
    }

    # model(image, conf=...) returns list of Results, filtering by conf like real YOLO
    def _inference(image, conf=0.5, verbose=False, **kwargs):
        result = _MockResult()
        mask = result.boxes.conf >= conf
        result.boxes.xyxy = result.boxes.xyxy[mask]
        result.boxes.conf = result.boxes.conf[mask]
        result.boxes.cls = result.boxes.cls[mask]
        return [result]

    model.side_effect = _inference
    return model


@pytest.fixture
def mock_yolo_model(monkeypatch):
    """Mock YOLO model that returns predictable detections.

    Returns 2 'person' detections and 1 'car' detection per frame.
    Avoids requiring actual YOLO weights download in CI.
    """
    model = _make_mock_yolo_model()

    # Patch ultralytics.YOLO so detector doesn't try to load real weights
    mock_yolo_class = MagicMock(return_value=model)
    monkeypatch.setattr(
        "dxd_vision.pipeline.detector.YOLO", mock_yolo_class, raising=False,
    )
    return model


@pytest.fixture
def detection_config():
    """DetectionConfig with test defaults (low confidence for testing)."""
    return DetectionConfig(
        enabled=True,
        model_path="yolov8n.pt",
        confidence_threshold=0.25,
        nms_threshold=0.45,
        target_classes=["person", "vehicle", "weapon", "package"],
        device="cpu",
        save_annotated_frame=True,
        save_detections_json=True,
    )


@pytest.fixture
def sample_frame_data(sample_video_path, default_config):
    """Single FrameData extracted from test video for detector testing."""
    from dxd_vision.pipeline.video_reader import VideoReader

    with VideoReader(sample_video_path, default_config.extraction) as reader:
        info = reader.get_video_info()
        for frame_num, image in reader.read_frames():
            return FrameData(
                image=image,
                metadata=FrameMetadata(
                    frame_number=frame_num,
                    timestamp_ms=0.0,
                    source_fps=info.fps,
                    extraction_fps=default_config.extraction.target_fps,
                    width=info.width,
                    height=info.height,
                    source_path=sample_video_path,
                ),
            )


@pytest.fixture
def processing_config():
    """DXDConfig with both extraction and detection enabled."""
    return DXDConfig(
        extraction=ExtractionConfig(
            target_fps=5.0,
            output_dir="./test_output",
            save_sample_frame=True,
        ),
        detection=DetectionConfig(
            enabled=True,
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            target_classes=["person", "vehicle", "weapon", "package"],
            device="cpu",
            save_annotated_frame=True,
            save_detections_json=True,
        ),
    )
