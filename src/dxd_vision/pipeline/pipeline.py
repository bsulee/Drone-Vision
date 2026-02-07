"""Vision pipeline orchestrator."""

import logging
from typing import Union

from dxd_vision.config.settings import DXDConfig
from dxd_vision.models.detection import (
    DetectionSummary,
    FrameDetections,
    ProcessingResult,
)
from dxd_vision.models.frame import ExtractionResult
from dxd_vision.models.tracking import (
    FrameTracking,
    TrackingResult,
)
from dxd_vision.pipeline.extractor import FrameExtractor
from dxd_vision.pipeline.video_reader import VideoReader

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrates video processing with three-way routing.

    Pipeline routing:
      tracking.enabled  → track mode  (detect + track in one pass)
      detection.enabled → detect mode (detect only)
      else              → extract only

    Thin wrapper that wires components together. Grows through Phases 2-7.
    """

    def __init__(self, config: DXDConfig):
        self._config = config

    def process_video(
        self, video_path: str
    ) -> Union[ExtractionResult, ProcessingResult, TrackingResult]:
        """Open video, extract frames, optionally detect/track, return results."""
        logger.info("Processing video: %s", video_path)

        with VideoReader(video_path, self._config.extraction) as reader:
            extractor = FrameExtractor(reader, self._config.extraction)
            video_info = reader.get_video_info()

            # Three-way routing: tracking > detection > extraction
            if self._config.tracking.enabled:
                return self._process_with_tracking(extractor, video_info)
            elif self._config.detection.enabled:
                return self._process_with_detection(extractor, video_info, video_path)
            else:
                result = extractor.extract_all()
                logger.info(
                    "Pipeline complete: %d frames extracted",
                    result.frames_extracted,
                )
                return result

    # ------------------------------------------------------------------
    # Tracking pipeline (Phase 3)
    # ------------------------------------------------------------------

    def _process_with_tracking(self, extractor, video_info):
        """Extract frames and run detection + tracking in a single pass."""
        from dxd_vision.pipeline.tracker import ObjectTracker

        tracker = ObjectTracker(self._config.detection, self._config.tracking)

        all_frame_trackings: list[FrameTracking] = []
        frames_extracted = 0
        sample_frame = None

        for frame in extractor.extract():
            if frames_extracted == 0 and self._config.extraction.save_sample_frame:
                sample_frame = frame
            frame_trk = tracker.track_frame(frame)
            all_frame_trackings.append(frame_trk)
            frames_extracted += 1

        # Save sample frame
        sample_path = None
        if sample_frame is not None:
            sample_path = extractor.save_sample_frame(
                sample_frame, self._config.extraction.output_dir
            )

        # Build tracking summary
        summary = tracker.build_summary(all_frame_trackings, frames_extracted)

        # Serialization
        tracking_path = None
        annotated_path = None
        if self._config.tracking.save_tracking_json:
            tracking_path = self._save_tracking_json(
                all_frame_trackings,
                tracker.build_trajectories(),
                self._config.extraction.output_dir,
            )
        if self._config.tracking.save_annotated_frame and sample_frame is not None:
            annotated_path = self._save_annotated_tracking_frame(
                sample_frame, all_frame_trackings, self._config.extraction.output_dir
            )

        target_fps = min(self._config.extraction.target_fps, video_info.fps)

        logger.info(
            "Pipeline complete: %d frames, %d unique objects tracked",
            frames_extracted,
            summary.total_unique_objects,
        )

        return TrackingResult(
            video_info=video_info,
            frames_extracted=frames_extracted,
            extraction_fps=target_fps,
            sample_frame_path=sample_path,
            tracking_enabled=True,
            tracking_summary=summary,
            tracking_path=tracking_path,
            annotated_frame_path=annotated_path,
        )

    @staticmethod
    def _save_tracking_json(frame_trackings, trajectories, output_dir: str) -> str:
        """Save all tracking data (per-frame + trajectories) to JSON."""
        import json
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "tracking.json"

        data = {
            "frames": [ft.model_dump() for ft in frame_trackings],
            "trajectories": [t.model_dump() for t in trajectories],
        }
        path.write_text(json.dumps(data, indent=2))

        logger.debug("Saved tracking JSON: %s", path)
        return str(path)

    @staticmethod
    def _save_annotated_tracking_frame(frame, frame_trackings, output_dir: str) -> str:
        """Draw bounding boxes with track IDs on the sample frame."""
        import cv2
        from pathlib import Path

        colors = {
            "weapon": (0, 0, 255),     # red
            "person": (0, 255, 255),   # yellow
            "vehicle": (0, 255, 0),    # green
            "package": (255, 0, 0),    # blue
        }

        annotated = frame.image.copy()

        for ft in frame_trackings:
            if ft.frame_number == frame.metadata.frame_number:
                for td in ft.tracked_detections:
                    color = colors.get(td.class_name, (255, 255, 255))
                    x, y = td.bbox.x, td.bbox.y
                    w, h = td.bbox.width, td.bbox.height
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    # Label includes object_id (e.g. "person_42 0.95")
                    label = f"{td.object_id} {td.confidence:.2f}"
                    cv2.putText(
                        annotated, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
                break

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "annotated_tracking.png"
        cv2.imwrite(str(path), annotated)

        logger.debug("Saved annotated tracking frame: %s", path)
        return str(path)

    # ------------------------------------------------------------------
    # Detection pipeline (Phase 2)
    # ------------------------------------------------------------------

    def _process_with_detection(self, extractor, video_info, video_path):
        """Extract frames and run YOLO detection in a single pass."""
        from dxd_vision.pipeline.detector import YOLODetector

        detector = YOLODetector(self._config.detection)

        all_frame_detections: list[FrameDetections] = []
        frames_extracted = 0
        sample_frame = None

        for frame in extractor.extract():
            if frames_extracted == 0 and self._config.extraction.save_sample_frame:
                sample_frame = frame
            frame_dets = detector.detect_frame(frame)
            all_frame_detections.append(frame_dets)
            frames_extracted += 1

        # Save sample frame
        sample_path = None
        if sample_frame is not None:
            sample_path = extractor.save_sample_frame(
                sample_frame, self._config.extraction.output_dir
            )

        # Build detection summary
        summary = self._build_summary(all_frame_detections, frames_extracted)

        # Serialization (bead ygc)
        detections_path = None
        annotated_path = None
        if self._config.detection.save_detections_json:
            detections_path = self._save_detections_json(
                all_frame_detections, self._config.extraction.output_dir
            )
        if self._config.detection.save_annotated_frame and sample_frame is not None:
            annotated_path = self._save_annotated_frame(
                sample_frame, all_frame_detections, self._config.extraction.output_dir
            )

        target_fps = min(self._config.extraction.target_fps, video_info.fps)

        logger.info(
            "Pipeline complete: %d frames, %d detections",
            frames_extracted,
            summary.total_detections,
        )

        return ProcessingResult(
            video_info=video_info,
            frames_extracted=frames_extracted,
            extraction_fps=target_fps,
            sample_frame_path=sample_path,
            detection_enabled=True,
            detection_summary=summary,
            detections_path=detections_path,
            annotated_frame_path=annotated_path,
        )

    @staticmethod
    def _build_summary(
        frame_detections: list[FrameDetections], total_frames: int
    ) -> DetectionSummary:
        """Aggregate per-frame detections into a summary."""
        by_class: dict[str, int] = {}
        total = 0
        confidence_sum = 0.0
        frames_with = 0

        for fd in frame_detections:
            if fd.count > 0:
                frames_with += 1
            for det in fd.detections:
                by_class[det.class_name] = by_class.get(det.class_name, 0) + 1
                confidence_sum += det.confidence
                total += 1

        return DetectionSummary(
            total_detections=total,
            by_class=by_class,
            avg_confidence=confidence_sum / total if total > 0 else 0.0,
            frames_with_detections=frames_with,
            frames_without_detections=total_frames - frames_with,
        )

    @staticmethod
    def _save_detections_json(
        frame_detections: list[FrameDetections], output_dir: str
    ) -> str:
        """Save all detections to a JSON file."""
        import json
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "detections.json"

        data = [fd.model_dump() for fd in frame_detections]
        path.write_text(json.dumps(data, indent=2))

        logger.debug("Saved detections JSON: %s", path)
        return str(path)

    @staticmethod
    def _save_annotated_frame(
        frame, frame_detections: list[FrameDetections], output_dir: str
    ) -> str:
        """Draw bounding boxes on the sample frame and save as PNG."""
        import cv2
        from pathlib import Path

        # Color map: class -> BGR color
        colors = {
            "weapon": (0, 0, 255),     # red
            "person": (0, 255, 255),   # yellow
            "vehicle": (0, 255, 0),    # green
            "package": (255, 0, 0),    # blue
        }

        annotated = frame.image.copy()

        # Find detections for this frame
        for fd in frame_detections:
            if fd.frame_number == frame.metadata.frame_number:
                for det in fd.detections:
                    color = colors.get(det.class_name, (255, 255, 255))
                    x, y = det.bbox.x, det.bbox.y
                    w, h = det.bbox.width, det.bbox.height
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    label = f"{det.class_name} {det.confidence:.2f}"
                    cv2.putText(
                        annotated, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
                break

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "annotated_sample.png"
        cv2.imwrite(str(path), annotated)

        logger.debug("Saved annotated frame: %s", path)
        return str(path)
