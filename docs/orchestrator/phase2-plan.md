# DXD Vision Engine — Phase 2: YOLO Object Detection

## Convoy Goal
**"Run YOLO detection on extracted frames, identify person/vehicle/weapon/package, and output structured detection results — proving the detect stage works before adding tracking."**

Success: `python -m dxd_vision --input video.mp4 --detect` produces detection results with bounding boxes and confidence scores.

---

## Architecture Extension

Phase 1 pipeline:
```
VideoReader → FrameExtractor → ExtractionResult
```

Phase 2 pipeline:
```
VideoReader → FrameExtractor → YOLODetector → ProcessingResult
                                    ↓
                              Iterator[FrameData] → Iterator[FrameDetections]
```

The `Iterator[FrameData]` interface from Phase 1 is consumed by the detector. Detection is optional — extraction-only mode still works.

---

## Beads Summary

| Bead | Title | Agent | Priority | Depends On | Merge Order |
|------|-------|-------|----------|------------|-------------|
| 1 | Merge frontend + define detection models/config | Implementation | P0 | None | 1 |
| 2 | Add ultralytics dependency + model download setup | Implementation | P0 | 1 | 2 |
| 3 | Final integration verification + v0.2.0-phase2 | Implementation | P1 | ALL | 11 (last) |
| 4 | Implement YOLODetector class | Back-End | P0 | 2 | 5 |
| 5 | Integrate detection into VisionPipeline | Back-End | P0 | 4 | 6 |
| 6 | Detection result serialization + annotated frames | Back-End | P1 | 5 | 7 |
| 7 | Add detection CLI options + update logging | Front-End | P0 | 1 | 3 |
| 8 | Implement detection summary and details display | Front-End | P1 | 1 | 4 |
| 9 | Create detection test fixtures + mock YOLO | Test | P0 | 2 | 8 |
| 10 | Write unit tests for YOLODetector | Test | P1 | 4, 9 | 9 |
| 11 | Write integration tests for detection pipeline | Test | P1 | 5, 7, 9 | 10 |

**Total Beads: 11**

---

## Dependency Graph

```
Bead 1 (merge frontend + detection models/config)
├── Bead 2 (ultralytics dependency + model setup)
│   ├── Bead 4 (YOLODetector)
│   │   ├── Bead 5 (pipeline integration)
│   │   │   ├── Bead 6 (serialization + annotated frames)
│   │   │   └── Bead 11 (integration tests)
│   │   └── Bead 10 (detector unit tests)
│   └── Bead 9 (test fixtures + mock YOLO)
│       ├── Bead 10 (detector unit tests)
│       └── Bead 11 (integration tests)
├── Bead 7 (CLI options + logging)
│   └── Bead 11 (integration tests)
├── Bead 8 (detection display)
└── Bead 3 (final integration) ← depends on ALL
```

## Parallel Work Tracks

After Bead 1 + 2:
- **Track A (Back-End):** Bead 4 → Bead 5 → Bead 6
- **Track B (Front-End):** Bead 7, Bead 8 (parallel, only need models)
- **Track C (Test):** Bead 9, then Beads 10 + 11 after Back-End lands

---

## Implementation Engineer

### Bead 1: Merge frontend/phase1 + define detection models/config

**Step 1:** Merge `frontend/phase1` branch to main (has CLI hardening + Phase 2 notes).

**Step 2:** Create new detection models.

**src/dxd_vision/models/detection.py:**
```python
from pydantic import BaseModel
from typing import Optional

class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x: int       # top-left x
    y: int       # top-left y
    width: int
    height: int

class Detection(BaseModel):
    """Single object detection."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    frame_number: int
    timestamp_ms: float

class FrameDetections(BaseModel):
    """All detections from a single frame."""
    frame_number: int
    timestamp_ms: float
    detections: list[Detection] = []

    @property
    def count(self) -> int:
        return len(self.detections)

class DetectionSummary(BaseModel):
    """Aggregated detection statistics."""
    total_detections: int
    by_class: dict[str, int]        # {"person": 23, "vehicle": 5}
    avg_confidence: float
    frames_with_detections: int
    frames_without_detections: int

class ProcessingResult(BaseModel):
    """Full pipeline result: extraction + optional detection."""
    video_info: "VideoInfo"
    frames_extracted: int
    extraction_fps: float
    sample_frame_path: Optional[str] = None
    detection_enabled: bool = False
    detection_summary: Optional[DetectionSummary] = None
    detections_path: Optional[str] = None       # JSON output path
    annotated_frame_path: Optional[str] = None   # sample with boxes
```

**Step 3:** Extend config.

**src/dxd_vision/config/settings.py** — add:
```python
class DetectionConfig(BaseModel):
    """Configuration for YOLO object detection."""
    enabled: bool = False
    model_path: str = "yolov8n.pt"   # default: YOLOv8 nano
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: list[str] = ["person", "vehicle", "weapon", "package"]
    device: str = "auto"             # "auto", "cpu", "cuda", "mps"
    save_annotated_frame: bool = True
    save_detections_json: bool = True

class DXDConfig(BaseModel):
    extraction: ExtractionConfig = ExtractionConfig()
    detection: DetectionConfig = DetectionConfig()   # NEW
```

**config/default.yaml** — add:
```yaml
detection:
  enabled: false
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  target_classes:
    - person
    - vehicle
    - weapon
    - package
  device: "auto"
  save_annotated_frame: true
  save_detections_json: true
```

**Acceptance:** All new models importable, serialize to JSON, config loads with detection section.

### Bead 2: Add ultralytics dependency + model download setup

- Add `ultralytics>=8.0` to requirements.txt
- Add download script or note: `yolo export model=yolov8n.pt` downloads automatically
- Verify `from ultralytics import YOLO` works
- Document GPU vs CPU detection in README

**Acceptance:** `pip install -r requirements.txt` includes ultralytics, YOLO importable.

### Bead 3: Final integration verification and Phase 2 release (DO THIS LAST)

- All branches merged to main
- `pytest tests/ -v` all green
- `python -m dxd_vision --input sample.mp4 --detect` works end-to-end
- Detection JSON saved to output dir
- Annotated sample frame with bounding boxes saved
- Tag `v0.2.0-phase2`

---

## Back-End Processing Engineer

### Bead 4: Implement YOLODetector class

**New file: src/dxd_vision/pipeline/detector.py:**
```python
from typing import Iterator
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, config: DetectionConfig):
        """Load YOLO model, configure thresholds."""
        self._config = config
        self._model = YOLO(config.model_path)
        # Map YOLO class names to our target classes
        self._class_map = self._build_class_map()

    def detect_frame(self, frame: FrameData) -> FrameDetections:
        """Run YOLO inference on a single frame."""
        # Run inference
        # Filter by confidence threshold
        # Filter by target classes
        # Apply NMS
        # Return FrameDetections

    def detect_stream(self, frames: Iterator[FrameData]) -> Iterator[FrameDetections]:
        """Run detection on a stream of frames."""
        for frame in frames:
            yield self.detect_frame(frame)

    def _build_class_map(self) -> dict:
        """Map YOLO COCO class IDs to our target class names."""
        # YOLO uses COCO classes: "person"=0, "car"=2, "truck"=7, "backpack"=24, etc.
        # Map these to our target: person, vehicle, weapon, package
```

**Key decisions:**
- YOLO COCO classes → our 4 target classes mapping (e.g., car+truck+bus = "vehicle")
- "weapon" won't be in standard COCO — handle gracefully (custom model later, or skip for Phase 2)
- Device auto-detection: CUDA > MPS > CPU

**Acceptance:** `YOLODetector(config).detect_frame(frame)` returns FrameDetections with valid detections.

### Bead 5: Integrate detection into VisionPipeline

**Extend src/dxd_vision/pipeline/pipeline.py:**
```python
class VisionPipeline:
    def process_video(self, video_path: str) -> ProcessingResult:
        """Extract frames and optionally run detection."""
        # Phase 1: extract frames
        # Phase 2: if detection enabled, run detector on extracted frames
        # Return ProcessingResult (superset of ExtractionResult)
```

**Critical:** Preserve Phase 1 extraction-only behavior when detection is disabled.

**Acceptance:** `VisionPipeline(config).process_video(path)` returns ProcessingResult. Detection runs when config.detection.enabled=True.

### Bead 6: Detection result serialization + annotated frames

- Save all detections to JSON file in output dir
- Save annotated frame (bounding boxes drawn on sample frame) as PNG
- JSON format matches the alert format from ORCHESTRATOR.md Phase 5 preview

**Acceptance:** JSON file with all detections written to output dir. Annotated PNG with visible bounding boxes saved.

---

## Front-End Design Engineer

### Bead 7: Add detection CLI options + update logging

**Extend src/dxd_vision/cli/main.py:**
```python
@click.option("--detect", "-d", is_flag=True, help="Enable YOLO object detection.")
@click.option("--model", "-m", default=None, help="YOLO model path (overrides config).")
@click.option("--confidence", default=None, type=float, help="Detection confidence threshold 0.0-1.0.")
@click.option("--classes", default=None, help="Comma-separated target classes.")
```

**Detection-specific logging:**
- "Loading YOLO model: yolov8n.pt (device: cuda)"
- "Running detection on 50 frames..."
- "Detection complete: 147 objects found in 38/50 frames"
- Timing: "Avg inference: 23ms/frame"

**Acceptance:** `python -m dxd_vision --input video.mp4 --detect` enables detection. `--help` shows all new options.

### Bead 8: Implement detection summary and details display

**Extend src/dxd_vision/cli/display.py:**
```python
class DisplayManager:
    # ... existing methods ...

    def show_detection_summary(self, summary: DetectionSummary) -> None:
        """Table with detection counts by class, avg confidence."""

    def show_detection_details(self, detections: list[FrameDetections], top_n: int = 10) -> None:
        """Verbose: show top N highest-confidence detections."""

    def show_processing_results(self, result: ProcessingResult) -> None:
        """Combined extraction + detection results."""
```

**Styling (from Phase 2 prep notes):**
- Threat-level colors: red=weapon, yellow=person, green=vehicle, blue=package
- No emoji — professional defense tables
- Narrow terminal support

**Acceptance:** Detection results display cleanly with class counts and color coding.

---

## Test Engineer

### Bead 9: Create detection test fixtures + mock YOLO model

**tests/conftest.py** — add fixtures:
```python
@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model that returns predictable detections."""
    # Returns 2 "person" detections and 1 "car" detection per frame

@pytest.fixture
def detection_config():
    """DetectionConfig with test defaults."""

@pytest.fixture
def sample_frame_data():
    """Single FrameData for detector testing."""
```

**Important:** Use mock/fake YOLO model for unit tests — don't require actual model weights download in CI.

**Acceptance:** Fixtures usable in all detection tests, mock model returns consistent detections.

### Bead 10: Write unit tests for YOLODetector

**tests/unit/test_detector.py — ~10 tests:**
- test_detector_initializes_with_config
- test_detect_frame_returns_frame_detections
- test_detect_frame_respects_confidence_threshold
- test_detect_frame_filters_target_classes
- test_detect_stream_yields_all_frames
- test_detect_stream_preserves_frame_order
- test_class_mapping_coco_to_target
- test_empty_frame_returns_zero_detections
- test_detection_bbox_within_frame_bounds
- test_detection_summary_aggregation

### Bead 11: Write integration tests for detection pipeline

**tests/integration/test_phase2_pipeline.py — ~5 tests:**
- test_end_to_end_extract_and_detect
- test_cli_with_detect_flag_exits_zero
- test_detection_json_output_written
- test_annotated_frame_saved
- test_detection_disabled_matches_phase1_behavior

---

## Integration Plan

### Merge Sequence
1. Bead 1 → main (merge frontend + detection models/config)
2. Bead 2 → main (ultralytics dependency)
3. Bead 7 → main (CLI detection options)
4. Bead 8 → main (detection display)
5. Bead 4 → main (YOLODetector)
6. Bead 5 → main (pipeline integration)
7. Bead 6 → main (serialization)
8. Bead 9 → main (test fixtures)
9. Bead 10 → main (detector tests)
10. Bead 11 → main (integration tests)
11. Bead 3 → main (final verification + tag)

### New Files
```
src/dxd_vision/
├── models/
│   └── detection.py        # NEW: Detection, FrameDetections, DetectionSummary, ProcessingResult
└── pipeline/
    └── detector.py          # NEW: YOLODetector class
tests/
├── unit/
│   └── test_detector.py     # NEW: YOLODetector unit tests
└── integration/
    └── test_phase2_pipeline.py  # NEW: detection integration tests
```

### Modified Files
```
src/dxd_vision/
├── config/settings.py       # Add DetectionConfig
├── pipeline/pipeline.py     # Extend with detection stage
├── cli/main.py              # Add --detect, --model, --confidence, --classes
├── cli/display.py           # Add detection display methods
├── utils/logging.py         # Detection-specific log messages
├── models/__init__.py       # Export new models
├── pipeline/__init__.py     # Export detector
config/default.yaml          # Add detection section
requirements.txt             # Add ultralytics
tests/conftest.py            # Add detection fixtures
```

---

## Phase Completion Criteria
- [ ] All 11 Beads closed
- [ ] `pytest tests/ -v` — all green (Phase 1 + Phase 2 tests)
- [ ] `python -m dxd_vision --input video.mp4 --detect` produces detections
- [ ] Detection JSON written to output dir
- [ ] Annotated frame with bounding boxes saved
- [ ] Phase 1 extraction-only mode still works (no regression)
- [ ] Git tag `v0.2.0-phase2` exists
- [ ] Pushed to GitHub

---

## YOLO Class Mapping (COCO → DXD Target)

| DXD Class | COCO Classes |
|-----------|-------------|
| person | person (0) |
| vehicle | car (2), motorcycle (3), bus (5), truck (7) |
| weapon | knife (43) — limited in COCO, custom model in future phases |
| package | backpack (24), suitcase (28), handbag (26) |

---

## Next Phase Preview (Phase 3: Object Tracking)
- Add DeepSORT/ByteTrack consuming detection results
- Assign persistent IDs to detected objects across frames
- Track trajectories and movement patterns
- New module: `src/dxd_vision/pipeline/tracker.py`
- Pipeline: extract → detect → track chain
