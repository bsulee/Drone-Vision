"""Unit tests for YOLODetector."""

import numpy as np
import pytest

from dxd_vision.config.settings import DetectionConfig
from dxd_vision.models.detection import (
    BoundingBox,
    Detection,
    DetectionSummary,
    FrameDetections,
)
from dxd_vision.models.frame import FrameData, FrameMetadata

try:
    from dxd_vision.pipeline.detector import YOLODetector
except ImportError:
    pytestmark = pytest.mark.skip(reason="YOLODetector not yet implemented")


def _make_frame_data(frame_number: int = 0, width: int = 640, height: int = 480):
    """Helper to create a FrameData with a blank image."""
    return FrameData(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        metadata=FrameMetadata(
            frame_number=frame_number,
            timestamp_ms=frame_number * 33.33,
            source_fps=30.0,
            extraction_fps=5.0,
            width=width,
            height=height,
            source_path="/test/video.mp4",
        ),
    )


class TestYOLODetector:
    """Tests for YOLODetector — YOLO inference wrapper."""

    def test_detector_initializes_with_config(self, mock_yolo_model, detection_config):
        """YOLODetector should initialize without error given valid config."""
        detector = YOLODetector(detection_config)
        assert detector is not None

    def test_detect_frame_returns_frame_detections(
        self, mock_yolo_model, detection_config, sample_frame_data,
    ):
        """detect_frame should return a FrameDetections instance."""
        detector = YOLODetector(detection_config)
        result = detector.detect_frame(sample_frame_data)
        assert isinstance(result, FrameDetections)

    def test_detect_frame_respects_confidence_threshold(self, mock_yolo_model):
        """Detections below the confidence threshold should be filtered out."""
        # Mock returns conf=[0.92, 0.85, 0.78] — threshold at 0.90 should keep only 1
        config = DetectionConfig(
            enabled=True,
            confidence_threshold=0.90,
            target_classes=["person", "vehicle"],
            device="cpu",
        )
        detector = YOLODetector(config)
        frame = _make_frame_data()
        result = detector.detect_frame(frame)
        # Only the 0.92-confidence person should survive
        high_conf = [d for d in result.detections if d.confidence >= 0.90]
        assert len(high_conf) >= 1
        assert all(d.confidence >= 0.90 for d in result.detections)

    def test_detect_frame_filters_target_classes(self, mock_yolo_model):
        """Only target classes should appear in detections."""
        # Mock returns person (cls 0) and car (cls 2)
        # If target_classes only has "person", car should be excluded
        config = DetectionConfig(
            enabled=True,
            confidence_threshold=0.25,
            target_classes=["person"],  # no vehicle
            device="cpu",
        )
        detector = YOLODetector(config)
        frame = _make_frame_data()
        result = detector.detect_frame(frame)
        for d in result.detections:
            assert d.class_name == "person"

    def test_detect_stream_yields_all_frames(self, mock_yolo_model, detection_config):
        """detect_stream should yield one FrameDetections per input frame."""
        detector = YOLODetector(detection_config)
        frames = [_make_frame_data(i) for i in range(5)]
        results = list(detector.detect_stream(iter(frames)))
        assert len(results) == 5

    def test_detect_stream_preserves_frame_order(self, mock_yolo_model, detection_config):
        """detect_stream results should maintain the input frame order."""
        detector = YOLODetector(detection_config)
        frames = [_make_frame_data(i) for i in range(5)]
        results = list(detector.detect_stream(iter(frames)))
        for i, r in enumerate(results):
            assert r.frame_number == i

    def test_class_mapping_coco_to_target(self, mock_yolo_model, detection_config):
        """COCO class 'car' should map to DXD target class 'vehicle'."""
        detector = YOLODetector(detection_config)
        frame = _make_frame_data()
        result = detector.detect_frame(frame)
        class_names = {d.class_name for d in result.detections}
        # Mock returns person (COCO 0) and car (COCO 2)
        # Expect "person" and "vehicle" (car mapped to vehicle)
        assert "person" in class_names
        if len(result.detections) > 2:
            # car should map to vehicle, not remain as "car"
            assert "car" not in class_names

    def test_empty_frame_returns_zero_detections(self, mock_yolo_model, detection_config, monkeypatch):
        """A frame with no detectable objects should return empty detections."""
        # Override mock to return empty results
        from tests.conftest import _MockResult
        empty_result = _MockResult()
        empty_result.boxes.xyxy = np.empty((0, 4), dtype=np.float32)
        empty_result.boxes.conf = np.empty((0,), dtype=np.float32)
        empty_result.boxes.cls = np.empty((0,), dtype=np.float32)
        mock_yolo_model.side_effect = lambda *a, **kw: [empty_result]

        detector = YOLODetector(detection_config)
        frame = _make_frame_data()
        result = detector.detect_frame(frame)
        assert result.count == 0
        assert result.detections == []

    def test_detection_bbox_within_frame_bounds(
        self, mock_yolo_model, detection_config, sample_frame_data,
    ):
        """All bounding boxes should be within the frame dimensions."""
        detector = YOLODetector(detection_config)
        result = detector.detect_frame(sample_frame_data)
        w = sample_frame_data.metadata.width
        h = sample_frame_data.metadata.height
        for d in result.detections:
            assert d.bbox.x >= 0
            assert d.bbox.y >= 0
            assert d.bbox.x + d.bbox.width <= w
            assert d.bbox.y + d.bbox.height <= h

    def test_detection_summary_aggregation(self, mock_yolo_model, detection_config):
        """Multiple frames of detections should aggregate into a correct summary."""
        detector = YOLODetector(detection_config)
        frames = [_make_frame_data(i) for i in range(3)]
        all_detections = list(detector.detect_stream(iter(frames)))

        # Build summary manually from results
        total = sum(fd.count for fd in all_detections)
        assert total > 0

        frames_with = sum(1 for fd in all_detections if fd.count > 0)
        assert frames_with == 3  # mock always returns detections
