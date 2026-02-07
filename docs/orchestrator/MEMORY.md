# DXD Vision Engine — Orchestrator Memory

## Current State
- **Project:** DXD Vision Engine (AI threat detection for Deus X Defense)
- **Phase 1: COMPLETE** — v0.1.0-phase1 tagged, 25/25 tests passing, 14 beads closed
- **Phase 2: COMPLETE** — v0.2.0-phase2 tagged, 40/40 tests passing, 11 beads closed
- **Phase 3: IN VERIFICATION** — Code written by orchestrator (mistake!), 11 verification beads created, agents about to launch
- **Key Files:**
  - `~/ORCHESTRATOR.md` — orchestrator system prompt (project desc, team, Beads workflow)
  - `~/phase1-plan.md` — Phase 1 plan (DONE)
  - `~/phase2-plan.md` — Phase 2 plan (DONE)
  - `~/agents/{implementation,backend,frontend,test}-engineer.md` — all agent prompts (UPDATED for Phase 3 verification)
  - `~/dxd-vision-engine/` — main repo (Implementation Engineer works here)
- **Git Worktrees (parallel development):**
  - `~/dxd-worktrees/backend/` → branch `backend/phase3`
  - `~/dxd-worktrees/frontend/` → branch `frontend/phase3`
  - `~/dxd-worktrees/test/` → branch `test/phase3`
- **GitHub:** `https://github.com/bsulee/Drone-Vision.git` — all branches pushed

## Phase 1 Summary (COMPLETE)
- 14 beads across 4 agents — all closed
- 25/25 tests passing (12 VideoReader + 9 Extractor + 4 integration)
- Core pipeline: `VideoReader → FrameExtractor → ExtractionResult`
- Core interface: `Iterator[FrameData]` carries through all phases
- Tagged `v0.1.0-phase1`

## Phase 2 Summary (COMPLETE)
- 11 beads across 4 agents — all closed
- 40/40 tests passing (Phase 1: 25 + Phase 2: 9 detector + 5 integration + 1 regression)
- Core new class: `YOLODetector` consuming `Iterator[FrameData]` → `Iterator[FrameDetections]`
- Pipeline: `extract → detect` chain (detect optional via config flag)
- Tagged `v0.2.0-phase2`

## Phase 3: Object Tracking (IN VERIFICATION)
- **Orchestrator wrote all code in single pass** (should have delegated — lesson learned!)
- 58/58 tests passing (40 Phase 1+2 + 12 tracker unit + 6 tracking integration)
- 11 verification beads created for agents to catch bugs
- Code committed to main, worktrees branched from it

### Phase 3 New Code
- `src/dxd_vision/models/tracking.py` — TrackedDetection, FrameTracking, ObjectTrajectory, TrackingSummary, TrackingResult
- `src/dxd_vision/pipeline/tracker.py` — ObjectTracker (wraps model.track(persist=True))
- Pipeline routing: tracking.enabled → track | detection.enabled → detect | else → extract
- CLI: --track, --tracker (bytetrack/botsort), --max-age
- Display: show_tracking_summary(), show_trajectories(), show_tracking_results()

### Phase 3 Verification Bead IDs
| Bead ID | Title | Agent |
|---------|-------|-------|
| `95j` | Verify tracking models + config | Implementation |
| `ae3` | Verify version bump + exports | Implementation |
| `1be` | Final integration + v0.3.0-phase3 | Implementation |
| `q7c` | Review ObjectTracker implementation | Back-End |
| `gsi` | Verify pipeline three-way routing | Back-End |
| `cik` | Verify tracking CLI options | Front-End |
| `teg` | Verify tracking display formatting | Front-End |
| `7g4` | Verify mock tracker fixtures | Test |
| `kmg` | Review + harden unit tests | Test |
| `ygw` | Review + harden integration tests | Test |
| `5k4` | Final QA gatekeeper | Test |

### Phase 3 Agent Launch Order
All agents can start immediately (code already exists):
1. **All 4 agents launch in parallel** — they're reviewing, not building
2. Test Engineer starts with `7g4` (blocks `kmg` and `ygw`)
3. Implementation Engineer does `1be` LAST (depends on all others)
4. Test Engineer does `5k4` as QA gatekeeper before `1be`

## Beads Syntax
```bash
bd create "Title" -d "Description" -p 1 -l label -a assignee --deps bd-xxx,bd-yyy
# Priority: 0-4 or P0-P4 (0=critical, 4=backlog). NOT words.
bd update <id> --status in_progress
bd close <id> --reason "what was done"
bd dep <blocker-id> --blocks <blocked-id>   # add dependency after creation
bd list --all     # include closed beads
```

## Team
- Implementation Engineer: models, config, deps, integration, merging
- Back-End Processing Engineer: VideoReader, FrameExtractor, YOLODetector, ObjectTracker, pipeline
- Front-End Design Engineer: CLI (Click), display (Rich), logging
- Test Engineer: fixtures, unit tests, integration tests, QA gatekeeper

## Architecture Decisions
- Generator/Iterator pipeline pattern (scales to Phase 7 RTSP)
- Pydantic models for data contracts
- YAML config with Pydantic validation (DXDConfig grows per phase)
- Beads for issue tracking
- COCO class mapping: person→person, car/truck/bus→vehicle, knife→weapon, backpack/suitcase→package
- Device auto-detection: CUDA > MPS > CPU
- ObjectTracker is STATEFUL (unlike stateless YOLODetector) — frames must arrive in order
- model.track(persist=True) combines detection + tracking in one pass (no separate detector needed)
- Object IDs: "{class}_{track_id}" (e.g., "person_42")

## User Preferences
- Terminal: tmux — scrollback is cleared by Claude Code TUI, user knows to read files
- Multi-line input: use `\` + Enter
- User wants files written for important output (can't rely on terminal scrollback)
- User is familiar with Anduril/Palantir patterns
- Deployment target: NVIDIA DGX Spark (1 PFLOP, 128GB RAM)
- **Agents run on Sonnet 4.5** (`/model sonnet` inside Claude Code)
- **Orchestrator runs on Opus 4.6**

## Git Worktree Setup
- Worktrees at `~/dxd-worktrees/{backend,frontend,test}`
- Phase 3 branches: `backend/phase3`, `frontend/phase3`, `test/phase3`
- `.claude/settings.local.json` already copied to each worktree
- Agents must run `export BEADS_NO_DAEMON=1` before any `bd` commands
- Main repo (`~/dxd-vision-engine/`) stays with Implementation Engineer on `main`
- **Orchestrator always opens in `~/dxd-vision-engine/`**

## Tmux Layout
```
┌──────────────────────┬──────────────────────┐
│    ORCHESTRATOR      │  BACK-END ENGINEER   │
│    (Opus 4.6)        │  (Sonnet 4.5)        │
├──────────────────────┼──────────────────────┤
│  IMPLEMENTATION      │  TEST ENGINEER       │
│  ENGINEER            │  (Sonnet 4.5)        │
│  (Sonnet 4.5)        ├──────────────────────┤
│                      │  FRONT-END ENGINEER  │
│                      │  (Sonnet 4.5)        │
└──────────────────────┴──────────────────────┘
```

## Lessons Learned
- **Beads priority is numeric 0-4, NOT words.**
- Write agent prompts to files (~/agents/) so user can launch in tmux panes.
- Claude Code TUI clears terminal scrollback — always write important output to files.
- **Always orient to `~/dxd-vision-engine/` first** — that's the project root.
- **`.claude/` doesn't propagate to worktrees** — must manually copy settings.local.json.
- **Verify worktree state with `git worktree list`** on restart before recreating.
- **Claude Code model selection is a slash command** — `/model sonnet` inside Claude Code, NOT `--model sonnet` CLI flag.
- **Agents tend to close beads without committing** — always remind them to commit + push before closing.
- **Test Engineer as integration gatekeeper works great** — run tests on merged main, triage failures.
- **For new phases, create new branches** (`backend/phase3`) and switch worktrees — don't reuse phase2 branches.
- **DON'T IMPLEMENT EVERYTHING YOURSELF.** The orchestrator's job is to delegate, not code. When one person writes everything, bugs hide because the same brain wrote both the code and the tests. Always delegate to agents and let them catch each other's mistakes.
- **`bd dep` command for adding deps after creation:** `bd dep <blocker-id> --blocks <blocked-id>` (not `bd update --deps`).
