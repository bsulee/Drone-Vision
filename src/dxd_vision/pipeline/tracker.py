"""Multi-object tracking using ultralytics built-in tracker.

ObjectTracker wraps model.track(persist=True), which combines detection
and tracking in a single pass.  It is STATEFUL — frames must be
processed in order so the tracker can maintain persistent IDs.
"""

import logging
from collections import defaultdict
from typing import Iterator

import torch
from ultralytics import YOLO

from dxd_vision.config.settings import DetectionConfig, TrackingConfig
from dxd_vision.models.detection import BoundingBox
from dxd_vision.models.frame import FrameData
from dxd_vision.models.tracking import (
    FrameTracking,
    ObjectTrajectory,
    TrackedDetection,
    TrackingSummary,
)

logger = logging.getLogger(__name__)

# Reuse the same COCO→DXD class mapping from detector.py
_COCO_TO_DXD: dict[str, str] = {
    "person": "person",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "knife": "weapon",
    "backpack": "package",
    "suitcase": "package",
    "handbag": "package",
}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ObjectTracker:
    """Multi-object tracker using ultralytics model.track(persist=True).

    Unlike YOLODetector (stateless), ObjectTracker is STATEFUL — it
    accumulates trajectory data and frames must arrive in temporal order.
    """

    def __init__(
        self,
        detection_config: DetectionConfig,
        tracking_config: TrackingConfig,
    ):
        self._det_config = detection_config
        self._trk_config = tracking_config
        self._device = _resolve_device(tracking_config.device)
        self._model = YOLO(detection_config.model_path)
        self._model.to(self._device)
        self._class_map = dict(_COCO_TO_DXD)

        # Trajectory accumulation: track_id -> list of (frame_number, class, conf, bbox)
        self._trajectories: dict[int, list[tuple[int, str, float, BoundingBox]]] = defaultdict(list)

        logger.info(
            "ObjectTracker ready: model=%s device=%s tracker=%s max_age=%d",
            detection_config.model_path,
            self._device,
            tracking_config.tracker,
            tracking_config.max_age,
        )

    def track_frame(self, frame: FrameData) -> FrameTracking:
        """Run detection + tracking on a single frame.

        Uses model.track(persist=True) which maintains internal state
        for cross-frame ID association.
        """
        results = self._model.track(
            frame.image,
            conf=self._det_config.confidence_threshold,
            iou=self._det_config.nms_threshold,
            persist=True,
            tracker=f"{self._trk_config.tracker}.yaml",
            verbose=False,
        )

        tracked: list[TrackedDetection] = []
        boxes = results[0].boxes

        # boxes.id can be None if no tracks established yet
        if boxes.id is not None:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                coco_name = self._model.names[class_id]
                dxd_class = self._class_map.get(coco_name)

                if dxd_class is None or dxd_class not in self._det_config.target_classes:
                    continue

                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                )
                confidence = float(box.conf[0])
                object_id = f"{dxd_class}_{track_id}"

                tracked.append(
                    TrackedDetection(
                        track_id=track_id,
                        object_id=object_id,
                        class_name=dxd_class,
                        confidence=confidence,
                        bbox=bbox,
                        frame_number=frame.metadata.frame_number,
                        timestamp_ms=frame.metadata.timestamp_ms,
                    )
                )

                # Accumulate trajectory
                self._trajectories[track_id].append(
                    (frame.metadata.frame_number, dxd_class, confidence, bbox)
                )

        return FrameTracking(
            frame_number=frame.metadata.frame_number,
            timestamp_ms=frame.metadata.timestamp_ms,
            tracked_detections=tracked,
        )

    def track_stream(
        self, frames: Iterator[FrameData]
    ) -> Iterator[FrameTracking]:
        """Run tracking on a stream of frames (must be in temporal order)."""
        for frame in frames:
            yield self.track_frame(frame)

    def build_trajectories(self) -> list[ObjectTrajectory]:
        """Build ObjectTrajectory objects from accumulated tracking data.

        Call this AFTER processing all frames.
        """
        trajectories: list[ObjectTrajectory] = []

        for track_id, entries in sorted(self._trajectories.items()):
            if not entries:
                continue

            frame_numbers = [e[0] for e in entries]
            classes = [e[1] for e in entries]
            confs = [e[2] for e in entries]
            bboxes = [e[3] for e in entries]

            # Use the most common class name for this track
            class_name = max(set(classes), key=classes.count)
            object_id = f"{class_name}_{track_id}"

            trajectories.append(
                ObjectTrajectory(
                    track_id=track_id,
                    object_id=object_id,
                    class_name=class_name,
                    first_frame=min(frame_numbers),
                    last_frame=max(frame_numbers),
                    total_frames=len(entries),
                    avg_confidence=sum(confs) / len(confs),
                    positions=bboxes,
                    frame_numbers=frame_numbers,
                )
            )

        return trajectories

    def build_summary(
        self,
        frame_trackings: list[FrameTracking],
        total_frames: int,
    ) -> TrackingSummary:
        """Build a TrackingSummary from frame tracking data and trajectories."""
        trajectories = self.build_trajectories()

        by_class: dict[str, int] = {}
        for traj in trajectories:
            by_class[traj.class_name] = by_class.get(traj.class_name, 0) + 1

        total_detections = sum(ft.count for ft in frame_trackings)
        frames_with = sum(1 for ft in frame_trackings if ft.count > 0)

        track_lengths = [t.total_frames for t in trajectories]

        return TrackingSummary(
            total_unique_objects=len(trajectories),
            by_class=by_class,
            avg_track_length=(
                sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
            ),
            longest_track=max(track_lengths) if track_lengths else 0,
            total_detections=total_detections,
            frames_with_tracks=frames_with,
            frames_without_tracks=total_frames - frames_with,
        )

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._trajectories.clear()
