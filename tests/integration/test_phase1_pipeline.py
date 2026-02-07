"""Integration tests for Phase 1 pipeline."""

import json
import subprocess
import sys

import pytest

from dxd_vision.config.settings import DXDConfig, ExtractionConfig
from dxd_vision.models.frame import ExtractionResult

try:
    from dxd_vision.pipeline.pipeline import VisionPipeline
except ImportError:
    pytestmark = pytest.mark.skip(reason="VisionPipeline not yet implemented")


class TestPhase1Pipeline:
    """End-to-end integration tests for the Phase 1 extraction pipeline."""

    def test_end_to_end_video_to_results(self, sample_video_path, default_config):
        """VisionPipeline should process a video and return ExtractionResult."""
        pipeline = VisionPipeline(default_config)
        result = pipeline.process_video(sample_video_path)
        assert isinstance(result, ExtractionResult)
        assert result.frames_extracted > 0
        assert result.video_info.total_frames == 300

    def test_cli_runs_without_error(self, sample_video_path, tmp_path):
        """Running the CLI via subprocess should exit 0."""
        result = subprocess.run(
            [
                sys.executable, "-m", "dxd_vision",
                "--input", sample_video_path,
                "--output", str(tmp_path / "cli_output"),
                "--fps", "5",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed with stderr:\n{result.stderr}"
        )

    def test_results_serializable_to_json(self, sample_video_path, default_config):
        """ExtractionResult should round-trip through JSON."""
        pipeline = VisionPipeline(default_config)
        result = pipeline.process_video(sample_video_path)

        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        restored = ExtractionResult(**parsed)

        assert restored.frames_extracted == result.frames_extracted
        assert restored.video_info.total_frames == result.video_info.total_frames
        assert restored.extraction_fps == result.extraction_fps

    def test_different_fps_targets(self, sample_video_path):
        """Pipeline should produce different frame counts for different FPS targets."""
        results = {}
        for target_fps in [1.0, 5.0, 10.0, 30.0]:
            config = DXDConfig(extraction=ExtractionConfig(
                target_fps=target_fps,
                save_sample_frame=False,
            ))
            pipeline = VisionPipeline(config)
            result = pipeline.process_video(sample_video_path)
            results[target_fps] = result.frames_extracted

        # More FPS = more frames extracted
        assert results[1.0] < results[5.0]
        assert results[5.0] < results[10.0]
        assert results[10.0] < results[30.0]

        # Rough accuracy checks
        assert abs(results[1.0] - 10) <= 1
        assert abs(results[5.0] - 50) <= 1
        assert abs(results[10.0] - 100) <= 1
        assert results[30.0] == 300
