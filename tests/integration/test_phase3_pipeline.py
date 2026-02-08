"""Integration tests for Phase 3 tracking pipeline."""

import json
import subprocess
import sys

import pytest

from dxd_vision.config.settings import DXDConfig, DetectionConfig, ExtractionConfig, TrackingConfig
from dxd_vision.models.tracking import TrackingResult
from dxd_vision.models.detection import ProcessingResult
from dxd_vision.models.frame import ExtractionResult

try:
    from dxd_vision.pipeline.tracker import ObjectTracker
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

try:
    from dxd_vision.pipeline.pipeline import VisionPipeline
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

pytestmark = pytest.mark.skipif(
    not (_HAS_TRACKER and _HAS_PIPELINE),
    reason="Tracking pipeline not yet implemented",
)


class TestPhase3Pipeline:
    """End-to-end integration tests for Phase 3 tracking pipeline."""

    def test_end_to_end_tracking(
        self, sample_video_path, tracking_pipeline_config, mock_tracking_model,
    ):
        """VisionPipeline with tracking enabled should return TrackingResult."""
        pipeline = VisionPipeline(tracking_pipeline_config)
        result = pipeline.process_video(sample_video_path)
        assert isinstance(result, TrackingResult)
        assert result.tracking_enabled is True
        assert result.tracking_summary is not None
        assert result.tracking_summary.total_unique_objects > 0

    def test_tracking_json_output_written(
        self, sample_video_path, tracking_pipeline_config, output_dir, mock_tracking_model,
    ):
        """Pipeline should write tracking results to a JSON file."""
        tracking_pipeline_config.extraction.output_dir = output_dir

        pipeline = VisionPipeline(tracking_pipeline_config)
        result = pipeline.process_video(sample_video_path)

        assert result.tracking_path is not None
        with open(result.tracking_path) as f:
            data = json.load(f)
        assert "frames" in data
        assert "trajectories" in data
        assert len(data["frames"]) > 0
        assert len(data["trajectories"]) > 0

    def test_annotated_tracking_frame_saved(
        self, sample_video_path, tracking_pipeline_config, output_dir, mock_tracking_model,
    ):
        """Pipeline should save an annotated frame with track IDs."""
        tracking_pipeline_config.extraction.output_dir = output_dir

        pipeline = VisionPipeline(tracking_pipeline_config)
        result = pipeline.process_video(sample_video_path)

        if result.annotated_frame_path is not None:
            import os
            assert os.path.exists(result.annotated_frame_path)
            assert result.annotated_frame_path.endswith(".png")

    def test_tracking_disabled_falls_through_to_detection(
        self, sample_video_path, processing_config, mock_yolo_model,
    ):
        """With tracking disabled but detection enabled, should return ProcessingResult."""
        pipeline = VisionPipeline(processing_config)
        result = pipeline.process_video(sample_video_path)
        assert isinstance(result, ProcessingResult)
        assert result.detection_enabled is True

    def test_both_disabled_matches_phase1_behavior(
        self, sample_video_path, default_config,
    ):
        """With tracking + detection disabled, pipeline should behave like Phase 1."""
        pipeline = VisionPipeline(default_config)
        result = pipeline.process_video(sample_video_path)
        assert result.frames_extracted > 0
        assert result.video_info.total_frames == 300

    def test_trajectory_data_correct(
        self, sample_video_path, tracking_pipeline_config, output_dir, mock_tracking_model,
    ):
        """Trajectory data in JSON should have valid structure."""
        tracking_pipeline_config.extraction.output_dir = output_dir

        pipeline = VisionPipeline(tracking_pipeline_config)
        result = pipeline.process_video(sample_video_path)

        with open(result.tracking_path) as f:
            data = json.load(f)

        # Check trajectory structure
        for traj in data["trajectories"]:
            assert "track_id" in traj
            assert "object_id" in traj
            assert "class_name" in traj
            assert "first_frame" in traj
            assert "last_frame" in traj
            assert "total_frames" in traj
            assert "positions" in traj
            assert "frame_numbers" in traj
            assert traj["total_frames"] == len(traj["positions"])
            assert traj["total_frames"] == len(traj["frame_numbers"])
            assert "_" in traj["object_id"]

    def test_cli_track_flag(self, sample_video_path, output_dir, mock_tracking_model):
        """CLI with --track flag should exit 0."""
        cmd = [
            sys.executable, "-m", "dxd_vision",
            "--input", str(sample_video_path),
            "--output", output_dir,
            "--track",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_botsort_tracker(self, sample_video_path, output_dir, mock_tracking_model):
        """CLI with --tracker botsort should work (not just bytetrack)."""
        cmd = [
            sys.executable, "-m", "dxd_vision",
            "--input", str(sample_video_path),
            "--output", output_dir,
            "--track",
            "--tracker", "botsort",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI with botsort failed: {result.stderr}"

    def test_cli_max_age_override(self, sample_video_path, output_dir, mock_tracking_model):
        """CLI with --max-age should override default."""
        cmd = [
            sys.executable, "-m", "dxd_vision",
            "--input", str(sample_video_path),
            "--output", output_dir,
            "--track",
            "--max-age", "10",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI with --max-age failed: {result.stderr}"

    def test_track_without_detect_flag(
        self, sample_video_path, tracking_pipeline_config, mock_tracking_model,
    ):
        """--track without --detect should auto-enable detection."""
        # Create config with detection disabled but tracking enabled
        config = DXDConfig(
            extraction=ExtractionConfig(
                target_fps=5.0,
                output_dir="./test_output",
            ),
            detection=DetectionConfig(
                enabled=False,  # explicitly disabled
                confidence_threshold=0.5,
                target_classes=["person", "vehicle"],
            ),
            tracking=TrackingConfig(
                enabled=True,  # but tracking enabled
                tracker="bytetrack",
            ),
        )
        # Pipeline should auto-enable detection since tracking requires it
        pipeline = VisionPipeline(config)
        result = pipeline.process_video(sample_video_path)
        assert isinstance(result, TrackingResult)
        assert result.tracking_enabled is True
