"""Video processing pipeline."""

from dxd_vision.pipeline.extractor import FrameExtractor
from dxd_vision.pipeline.pipeline import VisionPipeline
from dxd_vision.pipeline.video_reader import VideoReader

__all__ = ["FrameExtractor", "VideoReader", "VisionPipeline"]

try:
    from dxd_vision.pipeline.detector import YOLODetector
    __all__.append("YOLODetector")
except ImportError:
    pass  # torch/ultralytics not installed — extraction-only mode

try:
    from dxd_vision.pipeline.tracker import ObjectTracker
    __all__.append("ObjectTracker")
except ImportError:
    pass  # torch/ultralytics not installed — tracking unavailable
