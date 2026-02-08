# DXD Vision Engine — Fresh Start Guide

> **For Claude Code:** If you are a Claude Code agent reading this for the first time on a new machine, follow every step in order. This gets you from a blank Mac to a fully working multi-agent development environment.

---

## What Is This Project?

DXD Vision Engine is an AI-powered threat detection system for Deus X Defense. It processes video, detects objects (people, vehicles, weapons, packages) with YOLO, tracks them across frames with persistent IDs, and will eventually analyze behavior and push real-time alerts to a command dashboard.

**Architecture:** `VideoReader → FrameExtractor → [YOLODetector | ObjectTracker] → Results`

**7-Phase Plan:**
1. ~~Video Input + Frame Extraction~~ — **COMPLETE** (`v0.1.0-phase1`)
2. ~~YOLO Object Detection~~ — **COMPLETE** (`v0.2.0-phase2`)
3. Object Tracking — **CODE WRITTEN, UNDER VERIFICATION**
4. Behavior Analysis — planned
5. Alert Generation — planned
6. WebSocket Server — planned
7. Live RTSP Stream Input — planned

**Tech Stack:** Python 3.10+, OpenCV, Ultralytics (YOLOv8), Pydantic, Click, Rich, PyYAML, pytest

**Deployment Target:** NVIDIA DGX Spark (1 PFLOP, 128GB RAM)

---

## Part 1: Install Everything

Run these commands on the fresh Mac. Order matters.

### 1.1 Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Follow the instructions to add brew to PATH (it prints them at the end)
```

### 1.2 Core Tools
```bash
brew install git python@3.12 node tmux gh
```

### 1.3 GitHub Authentication
```bash
gh auth login
# Choose: GitHub.com → HTTPS → Login with browser
# This stores credentials in the macOS keychain
# Account: bsulee
```

### 1.4 Git Identity
```bash
git config --global user.name "Brian Sullivan"
git config --global user.email "your-email@example.com"
```

### 1.5 Claude Code
```bash
npm install -g @anthropic-ai/claude-code
# Then authenticate:
claude
# It will prompt you to log in via browser on first run. Do it, then exit.
```

### 1.6 Beads (Issue Tracker)
```bash
brew install beads
# Verify:
bd --version
# Should show: bd version 0.49.x
```

---

## Part 2: Clone and Set Up the Project

### 2.1 Clone the Repo
```bash
cd ~
git clone https://github.com/bsulee/Drone-Vision.git dxd-vision-engine
cd ~/dxd-vision-engine
```

### 2.2 Install Python Dependencies
```bash
cd ~/dxd-vision-engine
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python pydantic pyyaml rich click numpy ultralytics pytest
pip install -e .
```

> **Note:** `ultralytics` pulls in `torch` (~2GB). On Apple Silicon this includes MPS (GPU) support automatically. On the DGX Spark it will use CUDA.

### 2.3 Verify Tests Pass
```bash
cd ~/dxd-vision-engine
PYTHONPATH=src python3 -m pytest tests/ -v
# Expected: 58+ tests passed (Phase 1 + 2 + 3)
```

---

## Part 3: Restore Orchestrator Files

These files live in your home directory, NOT in the repo. Copies are bundled at `docs/orchestrator/` for portability.

```bash
# Orchestrator system prompt
cp ~/dxd-vision-engine/docs/orchestrator/ORCHESTRATOR.md ~/

# Phase plans
cp ~/dxd-vision-engine/docs/orchestrator/phase1-plan.md ~/
cp ~/dxd-vision-engine/docs/orchestrator/phase2-plan.md ~/

# Agent prompts (4 agents)
mkdir -p ~/agents
cp ~/dxd-vision-engine/docs/orchestrator/agents/*.md ~/agents/

# Claude Code memory for the orchestrator
# The path is based on your macOS username and the CWD you launch Claude from.
# If your username is "briansullivan" and you launch from ~/dxd-vision-engine:
mkdir -p ~/.claude/projects/-Users-$(whoami)/memory/
cp ~/dxd-vision-engine/docs/orchestrator/MEMORY.md \
   ~/.claude/projects/-Users-$(whoami)/memory/MEMORY.md
```

---

## Part 4: Set Up Git Worktrees (Multi-Agent Parallel Development)

Each agent works in its own git worktree on its own branch. This allows parallel development without conflicts.

```bash
mkdir -p ~/dxd-worktrees

cd ~/dxd-vision-engine
git worktree add ~/dxd-worktrees/backend backend/phase3
git worktree add ~/dxd-worktrees/frontend frontend/phase3
git worktree add ~/dxd-worktrees/test test/phase3

# Copy Claude Code permissions to each worktree
for wt in backend frontend test; do
  mkdir -p ~/dxd-worktrees/$wt/.claude
  cp ~/dxd-vision-engine/.claude/settings.local.json ~/dxd-worktrees/$wt/.claude/
done

# Verify:
git worktree list
# Should show 4 entries: main + 3 worktrees
```

---

## Part 5: Set Up Tmux Layout

The team uses 5 tmux panes — 1 orchestrator + 4 agents:

```
┌──────────────────────┬──────────────────────┐
│    ORCHESTRATOR      │  BACK-END ENGINEER   │
│    (Opus 4)          │  (Sonnet)            │
├──────────────────────┼──────────────────────┤
│  IMPLEMENTATION      │  TEST ENGINEER       │
│  ENGINEER            │  (Sonnet)            │
│  (Sonnet)            ├──────────────────────┤
│                      │  FRONT-END ENGINEER  │
│                      │  (Sonnet)            │
└──────────────────────┴──────────────────────┘
```

```bash
# Create session with the layout
tmux new-session -s dxd -d
tmux split-window -h -t dxd
tmux split-window -v -t dxd:0.0
tmux split-window -v -t dxd:0.1
tmux split-window -v -t dxd:0.3

# Set working directories
tmux send-keys -t dxd:0.0 'cd ~/dxd-vision-engine && clear' Enter
tmux send-keys -t dxd:0.1 'cd ~/dxd-vision-engine && clear' Enter
tmux send-keys -t dxd:0.2 'cd ~/dxd-worktrees/backend && clear' Enter
tmux send-keys -t dxd:0.3 'cd ~/dxd-worktrees/test && clear' Enter
tmux send-keys -t dxd:0.4 'cd ~/dxd-worktrees/frontend && clear' Enter

# Attach to the session
tmux attach -t dxd
```

### Launch Claude Code in Each Pane

From inside tmux, select each pane (Ctrl-B then arrow keys) and run:

| Pane | Role | Command | Then inside Claude |
|------|------|---------|--------------------|
| 0.0 | Orchestrator | `claude` | `/model opus` |
| 0.1 | Implementation Engineer | `claude` | `/model sonnet` |
| 0.2 | Back-End Engineer | `claude` | `/model sonnet` |
| 0.3 | Test Engineer | `claude` | `/model sonnet` |
| 0.4 | Front-End Engineer | `claude` | `/model sonnet` |

---

## Part 6: Current Project State (READ THIS)

### What Exists
- **58 tests** across 3 phases, all passing
- **Beads issue tracker** with full history of Phase 1 (14 beads), Phase 2 (11 beads), Phase 3 (11 beads)
- Run `export BEADS_NO_DAEMON=1 && bd list` to see open beads
- Run `export BEADS_NO_DAEMON=1 && bd list --all` to see everything

### Phase 3 Status
The orchestrator (incorrectly) wrote all Phase 3 code in a single pass instead of delegating to agents. The code is committed to main. 11 verification beads were created for agents to review and catch bugs. The Implementation Engineer's beads (`95j`, `ae3`) are already closed. Remaining open beads are for Back-End, Front-End, and Test engineers.

### Key Files
| File | Purpose |
|------|---------|
| `~/ORCHESTRATOR.md` | Orchestrator system prompt (project desc, team, beads workflow) |
| `~/agents/*.md` | Agent prompts for each engineer |
| `~/phase1-plan.md` | Phase 1 plan (DONE) |
| `~/phase2-plan.md` | Phase 2 plan (DONE) |
| `~/.claude/projects/-Users-*/memory/MEMORY.md` | Claude Code orchestrator memory |
| `src/dxd_vision/` | All source code |
| `tests/` | All tests |
| `config/default.yaml` | Default configuration |
| `.beads/` | Beads issue tracker database |

### Architecture
```
src/dxd_vision/
├── __init__.py          # version 0.3.0
├── cli/
│   ├── main.py          # Click CLI (--input, --detect, --track, etc.)
│   └── display.py       # Rich terminal display
├── config/
│   └── settings.py      # Pydantic config (Extraction, Detection, Tracking)
├── models/
│   ├── frame.py         # FrameData, VideoInfo, ExtractionResult
│   ├── detection.py     # Detection, FrameDetections, ProcessingResult
│   └── tracking.py      # TrackedDetection, FrameTracking, TrackingResult
├── pipeline/
│   ├── video_reader.py  # OpenCV video wrapper
│   ├── extractor.py     # FPS decimation, Iterator[FrameData]
│   ├── detector.py      # YOLODetector (stateless)
│   ├── tracker.py       # ObjectTracker (stateful, model.track)
│   └── pipeline.py      # VisionPipeline (three-way routing)
└── utils/
    └── logging.py       # Structured logging setup
```

### Pipeline Routing
```python
if config.tracking.enabled:    # --track flag
    # ObjectTracker: model.track(persist=True) → TrackingResult
elif config.detection.enabled:  # --detect flag
    # YOLODetector: model(image) → ProcessingResult
else:
    # FrameExtractor only → ExtractionResult
```

### Team & Workflow
- **Orchestrator** (Opus): Plans phases, creates beads, writes agent prompts, coordinates
- **Implementation Engineer** (Sonnet): Models, config, deps, merging, tagging — works on `main`
- **Back-End Engineer** (Sonnet): Pipeline code — works on `~/dxd-worktrees/backend/`
- **Front-End Engineer** (Sonnet): CLI + display — works on `~/dxd-worktrees/frontend/`
- **Test Engineer** (Sonnet): Tests + QA gatekeeper — works on `~/dxd-worktrees/test/`

All agents use `export BEADS_NO_DAEMON=1` before any `bd` commands (required in worktrees).

### Lessons Learned (Don't Repeat These)
1. **DON'T implement everything yourself.** Delegate to agents. That's the whole point.
2. **Beads priority is numeric 0-4**, not words.
3. **Agents tend to close beads without committing** — remind them to commit + push first.
4. **`.claude/` doesn't propagate to worktrees** — must manually copy `settings.local.json`.
5. **Claude Code model is a slash command** — `/model sonnet` inside Claude Code, NOT a CLI flag.
6. **Write important output to files** — Claude Code TUI clears terminal scrollback.
7. **Test Engineer as QA gatekeeper works great** — run tests on merged main, triage failures.
8. **For new phases, create new branches** and switch worktrees — don't reuse old branches.
9. **`bd dep <blocker> --blocks <blocked>`** to add dependencies after bead creation.
10. **When launching agents in tmux, use `tmux send-keys`** — don't use Task subagents.

---

## Part 7: What To Do Next

1. Check open beads: `export BEADS_NO_DAEMON=1 && bd list`
2. If Phase 3 verification beads are still open → finish them
3. If Phase 3 is done → tag `v0.3.0-phase3` and start planning Phase 4 (Behavior Analysis)
4. Phase 4 will need: behavior rules (loitering, fence breach, running, group formation, abandoned objects), trajectory analysis from Phase 3's `ObjectTrajectory` data

---

## Quick Smoke Test (Run After Setup)

```bash
cd ~/dxd-vision-engine
git log --oneline -5                                    # Recent history
git tag -l                                               # v0.1.0-phase1, v0.2.0-phase2
git worktree list                                        # 4 entries
export BEADS_NO_DAEMON=1 && bd list                     # Open beads
PYTHONPATH=src python3 -m pytest tests/ -v              # All green
PYTHONPATH=src python3 -m dxd_vision --help             # CLI works
```

If all of the above works, you're good to go.
