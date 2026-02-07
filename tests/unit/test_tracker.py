"""Unit tests for ObjectTracker."""

import numpy as np
import pytest

from dxd_vision.config.settings import DetectionConfig, TrackingConfig
from dxd_vision.models.tracking import (
    FrameTracking,
    ObjectTrajectory,
    TrackedDetection,
    TrackingSummary,
)
from dxd_vision.models.frame import FrameData, FrameMetadata

try:
    from dxd_vision.pipeline.tracker import ObjectTracker
except ImportError:
    pytestmark = pytest.mark.skip(reason="ObjectTracker not yet implemented")


def _make_frame_data(frame_number: int = 0, width: int = 640, height: int = 480):
    """Helper to create a FrameData with a blank image."""
    return FrameData(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        metadata=FrameMetadata(
            frame_number=frame_number,
            timestamp_ms=frame_number * 200.0,  # 5 fps
            source_fps=30.0,
            extraction_fps=5.0,
            width=width,
            height=height,
            source_path="/test/video.mp4",
        ),
    )


class TestObjectTracker:
    """Tests for ObjectTracker — multi-object tracking wrapper."""

    def test_tracker_initializes_with_config(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """ObjectTracker should initialize without error given valid config."""
        tracker = ObjectTracker(detection_config, tracking_config)
        assert tracker is not None

    def test_track_frame_returns_frame_tracking(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """track_frame should return a FrameTracking instance."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frame = _make_frame_data(0)
        result = tracker.track_frame(frame)
        assert isinstance(result, FrameTracking)

    def test_track_frame_has_track_ids(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """Each tracked detection should have a track_id and object_id."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frame = _make_frame_data(0)
        result = tracker.track_frame(frame)
        assert result.count > 0
        for td in result.tracked_detections:
            assert isinstance(td.track_id, int)
            assert td.track_id > 0
            assert "_" in td.object_id  # e.g. "person_1"

    def test_track_frame_object_id_format(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """object_id should follow '{class}_{track_id}' format."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frame = _make_frame_data(0)
        result = tracker.track_frame(frame)
        for td in result.tracked_detections:
            parts = td.object_id.split("_")
            assert len(parts) == 2
            assert parts[0] == td.class_name
            assert parts[1] == str(td.track_id)

    def test_track_stream_yields_all_frames(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """track_stream should yield one FrameTracking per input frame."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(5)]
        results = list(tracker.track_stream(iter(frames)))
        assert len(results) == 5

    def test_track_stream_preserves_frame_order(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """track_stream results should maintain the input frame order."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(5)]
        results = list(tracker.track_stream(iter(frames)))
        for i, r in enumerate(results):
            assert r.frame_number == i

    def test_persistent_ids_across_frames(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """The same object should keep the same track_id across multiple frames."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(3)]
        results = list(tracker.track_stream(iter(frames)))

        # Mock returns consistent IDs (1, 2, 3) across frames
        for result in results:
            track_ids = {td.track_id for td in result.tracked_detections}
            assert track_ids == {1, 2, 3}

    def test_class_mapping_coco_to_dxd(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """COCO class 'car' should map to DXD target class 'vehicle'."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frame = _make_frame_data(0)
        result = tracker.track_frame(frame)
        class_names = {td.class_name for td in result.tracked_detections}
        assert "person" in class_names
        assert "vehicle" in class_names
        assert "car" not in class_names

    def test_build_trajectories(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """build_trajectories should return one ObjectTrajectory per tracked object."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(5)]
        list(tracker.track_stream(iter(frames)))  # consume to accumulate

        trajectories = tracker.build_trajectories()
        assert len(trajectories) > 0

        for traj in trajectories:
            assert isinstance(traj, ObjectTrajectory)
            assert traj.total_frames == 5
            assert traj.first_frame == 0
            assert traj.last_frame == 4
            assert len(traj.positions) == 5
            assert len(traj.frame_numbers) == 5

    def test_build_summary(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """build_summary should produce correct aggregate statistics."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(5)]
        frame_trackings = list(tracker.track_stream(iter(frames)))

        summary = tracker.build_summary(frame_trackings, 5)
        assert isinstance(summary, TrackingSummary)
        # Mock returns 3 objects (2 person + 1 car/vehicle) per frame
        assert summary.total_unique_objects == 3
        assert "person" in summary.by_class
        assert "vehicle" in summary.by_class
        assert summary.by_class["person"] == 2
        assert summary.by_class["vehicle"] == 1
        assert summary.avg_track_length == 5.0
        assert summary.longest_track == 5
        assert summary.total_detections == 15  # 3 detections * 5 frames
        assert summary.frames_with_tracks == 5
        assert summary.frames_without_tracks == 0

    def test_reset_clears_trajectories(
        self, mock_tracking_model, detection_config, tracking_config,
    ):
        """reset() should clear all accumulated trajectory data."""
        tracker = ObjectTracker(detection_config, tracking_config)
        frames = [_make_frame_data(i) for i in range(3)]
        list(tracker.track_stream(iter(frames)))

        assert len(tracker.build_trajectories()) > 0
        tracker.reset()
        assert len(tracker.build_trajectories()) == 0

    def test_confidence_threshold_filtering(self, mock_tracking_model, tracking_config):
        """Detections below confidence threshold should be filtered out."""
        # Mock returns conf=[0.92, 0.85, 0.78] — threshold at 0.90 keeps fewer
        config = DetectionConfig(
            enabled=True,
            confidence_threshold=0.90,
            target_classes=["person", "vehicle"],
            device="cpu",
        )
        tracker = ObjectTracker(config, tracking_config)
        frame = _make_frame_data(0)
        result = tracker.track_frame(frame)
        for td in result.tracked_detections:
            assert td.confidence >= 0.90
