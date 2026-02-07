# DXD Vision Engine — Phase 1: Video Input + Frame Extraction

## Convoy Goal
**"Process any video file and extract frames at a configurable FPS with full metadata — proving the pipeline architecture works before we add AI."**

Success: `python -m dxd_vision --input video.mp4` produces a clean extraction result with sample frame saved.

---

## Beads Summary

| Bead | Title | Agent | Priority | Depends On | Merge Order |
|------|-------|-------|----------|------------|-------------|
| 1 | Initialize Git repo and project structure | Implementation | High | None | 1 |
| 2 | Define core data models and config system | Implementation | High | 1 | 2 |
| 3 | Final integration verification + release | Implementation | High | ALL | 13 (last) |
| 4 | Implement VideoReader class | Back-End | High | 2 | 6 |
| 5 | Implement FrameExtractor with FPS decimation | Back-End | High | 2, 4 | 7 |
| 6 | Build pipeline orchestrator skeleton | Back-End | Medium | 4, 5 | 8 |
| 7 | Implement CLI entry point with Click | Front-End | High | 1 | 4 |
| 8 | Implement progress display and results formatting | Front-End | Medium | 2 | 5 |
| 9 | Configure structured logging | Front-End | Medium | 1 | 3 |
| 10 | Create test infrastructure + synthetic video | Test | High | 1, 2 | 9 |
| 11 | Write unit tests for VideoReader | Test | High | 4, 10 | 10 |
| 12 | Write unit tests for FrameExtractor | Test | High | 5, 10 | 11 |
| 13 | Write integration tests for Phase 1 pipeline | Test | High | 6, 7, 10 | 12 |

**Total Beads: 13**

---

## Dependency Graph

```
Bead 1 (repo structure)
├── Bead 2 (models + config)
│   ├── Bead 4 (VideoReader)
│   │   ├── Bead 5 (FrameExtractor)
│   │   │   ├── Bead 6 (pipeline orchestrator)
│   │   │   └── Bead 12 (extractor tests)
│   │   └── Bead 11 (reader tests)
│   ├── Bead 8 (display/formatting)
│   └── Bead 10 (test infrastructure)
├── Bead 7 (CLI entry point)
├── Bead 9 (logging)
└── Bead 13 (integration tests) ← depends on 6, 7, 10
    └── Bead 3 (final integration) ← depends on ALL
```

## Parallel Work Tracks

After Bead 1:
- **Track A (Back-End):** Bead 2 → Bead 4 → Bead 5 → Bead 6
- **Track B (Front-End):** Bead 7, Bead 9 (no dependency on Bead 2)
- **Track C (Test):** Bead 10 starts once Bead 2 lands

After Back-End beads complete:
- Beads 11, 12 run in parallel (unit tests)
- Bead 13 waits for everything

---

## Implementation Engineer

### Bead 1: Initialize Git repository and project structure

**Directory structure:**
```
dxd-vision-engine/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── default.yaml
├── src/
│   └── dxd_vision/
│       ├── __init__.py
│       ├── __main__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── frame.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── exceptions.py
│       │   ├── video_reader.py
│       │   ├── extractor.py
│       │   └── pipeline.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── display.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       └── utils/
│           ├── __init__.py
│           └── logging.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── fixtures/
    │   └── generate_test_videos.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_video_reader.py
    │   └── test_extractor.py
    └── integration/
        ├── __init__.py
        └── test_phase1_pipeline.py
```

**Dependencies (requirements.txt):**
```
opencv-python>=4.8
pydantic>=2.0
pyyaml>=6.0
rich>=13.0
click>=8.1
pytest>=7.4
numpy>=1.24
```

**Acceptance:** `pip install -r requirements.txt` succeeds, all imports resolve, `bd list` works.

### Bead 2: Define core data models and configuration system

**src/dxd_vision/models/frame.py:**
```python
from pydantic import BaseModel
from typing import Optional
import numpy as np

class FrameMetadata(BaseModel):
    frame_number: int
    timestamp_ms: float
    source_fps: float
    extraction_fps: float
    width: int
    height: int
    source_path: str
    class Config:
        arbitrary_types_allowed = True

class FrameData:
    """Container for frame image + metadata. Not Pydantic (holds numpy array)."""
    def __init__(self, image: np.ndarray, metadata: FrameMetadata):
        self.image = image
        self.metadata = metadata

class VideoInfo(BaseModel):
    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    codec: str

class ExtractionResult(BaseModel):
    video_info: VideoInfo
    frames_extracted: int
    extraction_fps: float
    sample_frame_path: Optional[str] = None
```

**src/dxd_vision/config/settings.py:**
```python
from pydantic import BaseModel
import yaml

class ExtractionConfig(BaseModel):
    target_fps: float = 5.0
    output_dir: str = "./output"
    save_sample_frame: bool = True
    supported_formats: list[str] = [".mp4", ".mov", ".avi", ".mkv"]

class DXDConfig(BaseModel):
    extraction: ExtractionConfig = ExtractionConfig()

def load_config(path: str = "config/default.yaml") -> DXDConfig:
    """Load config from YAML, falling back to defaults."""
```

**config/default.yaml:**
```yaml
extraction:
  target_fps: 5.0
  output_dir: "./output"
  save_sample_frame: true
```

**Acceptance:** All models importable, serialize to JSON via `.model_dump_json()`, config loads from YAML with fallback defaults.

### Bead 3: Final integration verification and Phase 1 release

- Run full `pytest tests/ -v` — all green
- Verify `python -m dxd_vision --input sample.mp4` end-to-end
- Update README with Phase 1 usage
- Tag `v0.1.0-phase1`
- Verify all 12 other Beads are closed

---

## Back-End Processing Engineer

### Bead 4: Implement VideoReader class

**src/dxd_vision/pipeline/video_reader.py:**
```python
class VideoReader:
    def __init__(self, video_path: str, config: ExtractionConfig): ...
    def get_video_info(self) -> VideoInfo: ...
    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]: ...
    def release(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, ...): ...
```

**src/dxd_vision/pipeline/exceptions.py:**
```python
class VideoNotFoundError(Exception): ...
class UnsupportedFormatError(Exception): ...
class VideoReadError(Exception): ...
```

**Acceptance:** Opens mp4/mov/avi, returns accurate VideoInfo, iterates all frames, context manager releases resources, raises clear exceptions for invalid inputs.

### Bead 5: Implement FrameExtractor with FPS decimation

**src/dxd_vision/pipeline/extractor.py:**
```python
class FrameExtractor:
    def __init__(self, reader: VideoReader, config: ExtractionConfig): ...
    def extract(self) -> Iterator[FrameData]: ...
    def extract_all(self) -> ExtractionResult: ...
    def save_sample_frame(self, frame: FrameData, output_dir: str) -> str: ...
```

**Critical:** Use timestamp-based frame selection, NOT simple modulo. The `extract() -> Iterator[FrameData]` pattern is the core pipeline interface — Phase 2 YOLO consumes it, Phase 7 RTSP provides it.

**Acceptance:** 30fps video at 5fps target yields `duration * 5` frames (+-1). Correct metadata. Sample PNG saves.

### Bead 6: Build pipeline orchestrator skeleton

**src/dxd_vision/pipeline/pipeline.py:**
```python
class VisionPipeline:
    def __init__(self, config: DXDConfig): ...
    def process_video(self, video_path: str) -> ExtractionResult: ...
```

Thin wrapper wiring VideoReader + FrameExtractor. Grows through Phases 2-7.

**Acceptance:** `VisionPipeline(config).process_video(path)` returns valid ExtractionResult.

---

## Front-End Design Engineer

### Bead 7: Implement CLI entry point with Click

**src/dxd_vision/cli/main.py:**
```python
@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True))
@click.option("--config", "-c", default="config/default.yaml")
@click.option("--output", "-o", default="./output")
@click.option("--fps", default=None, type=float)
@click.option("--verbose", "-v", is_flag=True)
def main(input, config, output, fps, verbose): ...
```

**src/dxd_vision/__main__.py:** Enables `python -m dxd_vision`.

**Acceptance:** `python -m dxd_vision --input video.mp4` works, `--help` clean, invalid paths show helpful errors.

### Bead 8: Implement progress display and results formatting

**src/dxd_vision/cli/display.py:**
```python
class DisplayManager:
    def show_header(self): ...
    def show_video_info(self, info: VideoInfo): ...
    def create_progress(self, total_frames: int) -> Progress: ...
    def show_results(self, result: ExtractionResult): ...
    def show_error(self, message: str): ...
```

Professional defense-product styling. Clean tables, progress bars, no emoji.

**Acceptance:** Video info in table, progress bar during extraction, results summary, no crashes on narrow terminals.

### Bead 9: Configure structured logging

**src/dxd_vision/utils/logging.py:**
```python
def setup_logging(verbose: bool = False) -> logging.Logger: ...
```

Format: `[2026-02-06 14:30:22] [INFO] [extractor] Extracting frames at 5.0 FPS`

**Acceptance:** Consistent logging across all modules, verbose flag works, no duplicate entries.

---

## Test Engineer

### Bead 10: Create test infrastructure and synthetic test video

**tests/conftest.py:**
```python
@pytest.fixture
def sample_video_path(tmp_path):
    """10-second, 30fps, 640x480 test video with frame numbers burned in."""

@pytest.fixture
def short_video_path(tmp_path):
    """2-second test video for quick tests."""

@pytest.fixture
def default_config(): ...

@pytest.fixture
def output_dir(tmp_path): ...
```

**Acceptance:** Fixtures create playable mp4, sample video has exactly 300 frames.

### Bead 11: Write unit tests for VideoReader

**tests/unit/test_video_reader.py — 12 tests:**
- test_opens_valid_mp4
- test_returns_correct_video_info
- test_video_info_frame_count (300)
- test_video_info_fps (30.0)
- test_video_info_dimensions (640x480)
- test_video_info_duration (~10s)
- test_read_frames_yields_all (300)
- test_read_frames_correct_shape (480x640x3)
- test_context_manager_releases
- test_file_not_found_raises
- test_unsupported_format_raises
- test_corrupt_video_raises

### Bead 12: Write unit tests for FrameExtractor

**tests/unit/test_extractor.py — 9 tests:**
- test_extract_at_5fps_from_30fps (50 frames)
- test_extract_at_1fps (10 frames)
- test_extract_at_source_fps (300 frames)
- test_frame_metadata_timestamps (monotonic)
- test_frame_metadata_dimensions (640x480)
- test_frame_metadata_source_path
- test_save_sample_frame (PNG exists)
- test_extract_all_returns_result
- test_extract_all_frame_count_matches

### Bead 13: Write integration tests for Phase 1 pipeline

**tests/integration/test_phase1_pipeline.py — 4 tests:**
- test_end_to_end_video_to_results
- test_cli_runs_without_error (subprocess, exit code 0)
- test_results_serializable_to_json
- test_different_fps_targets (1/5/10/30 fps)

---

## Integration Plan

### Merge Sequence
1. Bead 1 → main (repo scaffolding)
2. Bead 2 → main (models + config)
3. Bead 9 → main (logging)
4. Bead 7 → main (CLI skeleton)
5. Bead 8 → main (display formatting)
6. Bead 4 → main (VideoReader)
7. Bead 5 → main (FrameExtractor)
8. Bead 6 → main (pipeline orchestrator)
9. Bead 10 → main (test infrastructure)
10. Bead 11 → main (VideoReader tests)
11. Bead 12 → main (FrameExtractor tests)
12. Bead 13 → main (integration tests)
13. Bead 3 → main (final verification + tag)

### Integration Points
| Connection | From | To | Interface |
|------------|------|----|-----------|
| CLI → Pipeline | cli/main.py | pipeline/pipeline.py | `VisionPipeline.process_video(path) → ExtractionResult` |
| Pipeline → Reader | pipeline/pipeline.py | pipeline/video_reader.py | `VideoReader(path, config)` |
| Pipeline → Extractor | pipeline/pipeline.py | pipeline/extractor.py | `FrameExtractor(reader, config).extract() → Iterator[FrameData]` |
| CLI → Display | cli/main.py | cli/display.py | `DisplayManager.show_*()` |
| All → Config | * | config/settings.py | `load_config(path) → DXDConfig` |
| All → Logging | * | utils/logging.py | `setup_logging(verbose)` |

---

## Phase Completion Criteria
- [ ] All 13 Beads closed (`bd list` shows all closed)
- [ ] `pytest tests/ -v` — all green, 0 failures
- [ ] `python -m dxd_vision --input test_video.mp4` produces correct output
- [ ] Sample frame saved as readable PNG
- [ ] ExtractionResult JSON matches expected schema
- [ ] 30fps video at 5fps target yields correct frame count
- [ ] No resource leaks (OpenCV capture released)
- [ ] Git tag `v0.1.0-phase1` exists

---

## Next Phase Preview (Phase 2: YOLO Object Detection)
- Add YOLOv8/v11 consuming `Iterator[FrameData]`
- New model: `DetectionResult` (bounding boxes, class labels, confidence)
- New module: `src/dxd_vision/pipeline/detector.py`
- Pipeline extends: `extract() → detect()` chain
- Config adds: `detection:` section
- Estimated Beads: ~10

## Beads Commands

```bash
# Phase 1 Beads Setup
# Priority: 0=critical, 1=high, 2=normal, 3=low, 4=backlog
# Run from ~/dxd-vision-engine/

# Implementation Engineer
bd create "Initialize project directory structure" -d "Create src/dxd_vision/ scaffolding, tests/, requirements.txt, pyproject.toml, .gitignore, all __init__.py files" -p 0 -l phase-1 -a implementation
bd create "Define core data models and config system" -d "Pydantic models (FrameMetadata, FrameData, VideoInfo, ExtractionResult) and YAML config (DXDConfig, ExtractionConfig, load_config)" -p 0 -l phase-1 -a implementation --deps BEAD_1_ID
bd create "Final integration verification and Phase 1 release" -d "Run full test suite, verify CLI end-to-end, update README, tag v0.1.0-phase1" -p 1 -l phase-1 -a implementation --deps ALL_OTHER_IDS

# Back-End Processing Engineer
bd create "Implement VideoReader class" -d "OpenCV wrapper: get_video_info()->VideoInfo, read_frames()->Iterator, context manager, exceptions" -p 0 -l phase-1 -a backend --deps BEAD_2_ID
bd create "Implement FrameExtractor with FPS decimation" -d "Timestamp-based frame selection, extract()->Iterator[FrameData], save_sample_frame()" -p 0 -l phase-1 -a backend --deps BEAD_2_ID,BEAD_4_ID
bd create "Build pipeline orchestrator skeleton" -d "VisionPipeline.process_video(path)->ExtractionResult, wires VideoReader+FrameExtractor" -p 2 -l phase-1 -a backend --deps BEAD_4_ID,BEAD_5_ID

# Front-End Design Engineer
bd create "Implement CLI entry point with Click" -d "Click command: --input, --config, --output, --fps, --verbose. python -m dxd_vision support" -p 0 -l phase-1 -a frontend --deps BEAD_1_ID
bd create "Implement progress display and results formatting" -d "Rich library DisplayManager: show_header, show_video_info, create_progress, show_results" -p 2 -l phase-1 -a frontend --deps BEAD_2_ID
bd create "Configure structured logging" -d "setup_logging(verbose) for all modules, consistent format, no duplicate entries" -p 2 -l phase-1 -a frontend --deps BEAD_1_ID

# Test Engineer
bd create "Create test infrastructure and synthetic test video" -d "conftest.py fixtures: sample_video_path (10s/30fps/640x480), short_video_path, default_config" -p 0 -l phase-1 -a test --deps BEAD_1_ID,BEAD_2_ID
bd create "Write unit tests for VideoReader" -d "12 tests: happy path, video info accuracy, frame iteration, error cases" -p 1 -l phase-1 -a test --deps BEAD_4_ID,BEAD_10_ID
bd create "Write unit tests for FrameExtractor" -d "9 tests: FPS decimation, metadata, sample frame, extract_all" -p 1 -l phase-1 -a test --deps BEAD_5_ID,BEAD_10_ID
bd create "Write integration tests for Phase 1 pipeline" -d "4 tests: end-to-end pipeline, CLI subprocess, JSON serialization, multiple FPS targets" -p 1 -l phase-1 -a test --deps BEAD_6_ID,BEAD_7_ID,BEAD_10_ID
```

*Replace BEAD_*_ID placeholders with actual Bead IDs as they are created.*

*Replace BEAD_*_ID placeholders with actual Bead IDs as they are created.*
