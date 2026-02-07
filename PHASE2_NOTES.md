# Phase 2 - YOLO Detection: Front-End Changes

## From ORCHESTRATOR.md Phase 2:
- Integrate YOLOv8/YOLOv11 model
- Run detection on extracted frames
- Identify: person, vehicle, weapon, package
- Output bounding boxes with confidence scores
- Store detection results in JSON

## CLI Changes Needed (cli/main.py):

### New Options (Optional):
```python
@click.option("--model", "-m", default=None, help="Path to YOLO model weights (overrides config)")
@click.option("--confidence", default=None, type=float, help="Detection confidence threshold (0.0-1.0)")
@click.option("--classes", default=None, help="Comma-separated class names to detect (e.g., 'person,vehicle')")
```

### Config Updates:
- Add `DetectionConfig` to settings.py with:
  - model_path: str (path to YOLO weights)
  - confidence_threshold: float = 0.5
  - target_classes: list[str] = ["person", "vehicle", "weapon", "package"]
  - nms_threshold: float = 0.45 (non-max suppression)

### Validation:
- Validate confidence is between 0.0 and 1.0
- Ensure model file exists if specified

## Display Changes Needed (cli/display.py):

### New Method: `show_detection_summary()`
Display detection results table:
```
┌─────────────────────────────────┐
│     Detection Summary          │
├──────────────────┬──────────────┤
│ Class            │ Count        │
├──────────────────┼──────────────┤
│ person           │ 23           │
│ vehicle          │ 5            │
│ weapon           │ 0            │
│ package          │ 2            │
├──────────────────┼──────────────┤
│ Total Detections │ 30           │
│ Avg Confidence   │ 0.87         │
└──────────────────┴──────────────┘
```

### Enhanced `show_results()`:
Update ExtractionResult display to include:
- Detection counts by class
- Average confidence score
- High-confidence detections (> 0.9)
- Frames with detections vs. empty frames

### New Method: `show_detection_details()` (verbose mode):
Show top N detections with:
- Frame number
- Class name
- Confidence score
- Bounding box coordinates (optional)

## Data Model Changes (Already handled by Back-End):

The Back-End will extend models/frame.py with:
```python
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    frame_number: int

class DetectionResult(BaseModel):
    detections: list[Detection]
    total_count: int
    by_class: dict[str, int]
    avg_confidence: float
```

## Testing Considerations:
- Narrow terminal handling for new tables
- Long class names truncation
- Large detection counts (100+)
- Zero detections (graceful "No threats detected" message)

## Professional Defense Styling:
- Use color coding:
  - Green: Low-risk classes (vehicle)
  - Yellow: Medium-risk (person)
  - Red: High-risk (weapon)
- No emoji, professional tables only
- Clear threat level indicators
