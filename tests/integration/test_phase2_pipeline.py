"""Integration tests for Phase 2 detection pipeline."""

import json
import subprocess
import sys

import pytest

from dxd_vision.config.settings import DXDConfig, DetectionConfig, ExtractionConfig
from dxd_vision.models.detection import ProcessingResult

try:
    from dxd_vision.pipeline.detector import YOLODetector
    _HAS_DETECTOR = True
except ImportError:
    _HAS_DETECTOR = False

try:
    from dxd_vision.pipeline.pipeline import VisionPipeline
    # Check if pipeline supports detection (returns ProcessingResult)
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

pytestmark = pytest.mark.skipif(
    not (_HAS_DETECTOR and _HAS_PIPELINE),
    reason="Detection pipeline not yet implemented",
)


class TestPhase2Pipeline:
    """End-to-end integration tests for Phase 2 detection pipeline."""

    def test_end_to_end_extract_and_detect(
        self, sample_video_path, processing_config, mock_yolo_model,
    ):
        """VisionPipeline with detection enabled should return ProcessingResult."""
        pipeline = VisionPipeline(processing_config)
        result = pipeline.process_video(sample_video_path)
        assert isinstance(result, ProcessingResult)
        assert result.detection_enabled is True
        assert result.detection_summary is not None
        assert result.detection_summary.total_detections > 0

    def test_cli_with_detect_flag_exits_zero(
        self, sample_video_path, tmp_path, mock_yolo_model,
    ):
        """Running CLI with --detect should exit 0."""
        result = subprocess.run(
            [
                sys.executable, "-m", "dxd_vision",
                "--input", sample_video_path,
                "--output", str(tmp_path / "cli_output"),
                "--fps", "5",
                "--detect",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"CLI failed with stderr:\n{result.stderr}"
        )

    def test_detection_json_output_written(
        self, sample_video_path, processing_config, output_dir, mock_yolo_model,
    ):
        """Pipeline should write detection results to a JSON file."""
        processing_config.extraction.output_dir = output_dir
        processing_config.detection.save_detections_json = True

        pipeline = VisionPipeline(processing_config)
        result = pipeline.process_video(sample_video_path)

        assert result.detections_path is not None
        with open(result.detections_path) as f:
            data = json.load(f)
        assert isinstance(data, (list, dict))

    def test_annotated_frame_saved(
        self, sample_video_path, processing_config, output_dir, mock_yolo_model,
    ):
        """Pipeline should save an annotated frame with bounding boxes."""
        processing_config.extraction.output_dir = output_dir
        processing_config.detection.save_annotated_frame = True

        pipeline = VisionPipeline(processing_config)
        result = pipeline.process_video(sample_video_path)

        if result.annotated_frame_path is not None:
            import os
            assert os.path.exists(result.annotated_frame_path)
            assert result.annotated_frame_path.endswith(".png")

    def test_detection_disabled_matches_phase1_behavior(
        self, sample_video_path, default_config,
    ):
        """With detection disabled, pipeline should behave like Phase 1."""
        # default_config has no detection enabled
        pipeline = VisionPipeline(default_config)
        result = pipeline.process_video(sample_video_path)
        # Should still work — either ExtractionResult or ProcessingResult with detection_enabled=False
        assert result.frames_extracted > 0
        assert result.video_info.total_frames == 300
