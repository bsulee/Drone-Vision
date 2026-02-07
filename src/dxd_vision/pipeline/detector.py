"""YOLO object detection on video frames."""

import logging
from typing import Iterator

import torch
from ultralytics import YOLO

from dxd_vision.config.settings import DetectionConfig
from dxd_vision.models.detection import BoundingBox, Detection, FrameDetections
from dxd_vision.models.frame import FrameData

logger = logging.getLogger(__name__)

# COCO class names → DXD target class mapping
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


class YOLODetector:
    """Runs YOLO inference on FrameData, yielding FrameDetections."""

    def __init__(self, config: DetectionConfig):
        self._config = config
        self._device = _resolve_device(config.device)
        self._model = YOLO(config.model_path)
        self._model.to(self._device)
        self._class_map = dict(_COCO_TO_DXD)
        logger.info(
            "YOLODetector ready: model=%s device=%s conf=%.2f",
            config.model_path,
            self._device,
            config.confidence_threshold,
        )

    def detect_frame(self, frame: FrameData) -> FrameDetections:
        """Run YOLO inference on a single frame, return filtered detections."""
        results = self._model(
            frame.image,
            conf=self._config.confidence_threshold,
            iou=self._config.nms_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            coco_name = self._model.names[class_id]
            dxd_class = self._class_map.get(coco_name)

            if dxd_class is None or dxd_class not in self._config.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=dxd_class,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    ),
                    frame_number=frame.metadata.frame_number,
                    timestamp_ms=frame.metadata.timestamp_ms,
                )
            )

        return FrameDetections(
            frame_number=frame.metadata.frame_number,
            timestamp_ms=frame.metadata.timestamp_ms,
            detections=detections,
        )

    def detect_stream(
        self, frames: Iterator[FrameData]
    ) -> Iterator[FrameDetections]:
        """Run detection on a stream of frames."""
        for frame in frames:
            yield self.detect_frame(frame)
