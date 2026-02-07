# DXD Vision Engine — New Mac Setup Guide

## What Lives Where

**In the GitHub repo** (`https://github.com/bsulee/Drone-Vision.git`):
- All source code, tests, config
- `.beads/` — issue tracker database (git-tracked)
- `.claude/settings.local.json` — Claude Code permissions

**NOT in the repo** (home directory — must copy manually):
- `~/ORCHESTRATOR.md` — orchestrator system prompt
- `~/agents/*.md` — 4 agent prompts (implementation, backend, frontend, test)
- `~/phase1-plan.md`, `~/phase2-plan.md` — phase plans
- `~/.claude/projects/-Users-briansullivan/memory/MEMORY.md` — Claude Code orchestrator memory

---

## Step-by-Step Setup

### 1. Install Prerequisites

```bash
# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.10+ (the project needs this, current mac has 3.9)
brew install python@3.12

# Git (usually pre-installed on macOS)
brew install git

# tmux (for multi-agent layout)
brew install tmux
```

### 2. Install Claude Code

```bash
# Claude Code CLI
npm install -g @anthropic-ai/claude-code
# OR if npm not installed:
brew install node
npm install -g @anthropic-ai/claude-code
```

### 3. Install Beads (issue tracker)

```bash
# Check https://github.com/beads-project/beads for latest install method
brew install beads
# OR:
go install github.com/beads-project/beads/cmd/bd@latest
```

### 4. Clone the Repo

```bash
cd ~
git clone https://github.com/bsulee/Drone-Vision.git dxd-vision-engine
cd dxd-vision-engine
```

### 5. Install Python Dependencies

```bash
cd ~/dxd-vision-engine
pip3 install opencv-python pydantic pyyaml rich click numpy ultralytics pytest
# OR with a venv (recommended):
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python pydantic pyyaml rich click numpy ultralytics pytest
pip install -e .
```

### 6. Verify It Works

```bash
cd ~/dxd-vision-engine
PYTHONPATH=src python3 -m pytest tests/ -v
# Should see 58+ passed (Phase 1 + 2 + 3)
```

### 7. Copy Home Directory Files

These files are NOT in the repo. Copy them from your old Mac (AirDrop, USB, scp, etc):

```bash
# From OLD Mac, run:
scp ~/ORCHESTRATOR.md NEW_MAC:~/
scp ~/phase1-plan.md NEW_MAC:~/
scp ~/phase2-plan.md NEW_MAC:~/
scp -r ~/agents NEW_MAC:~/

# Claude Code memory (path will differ on new Mac — see step 8)
```

**OR** — I've bundled copies into the repo (see step 7b below).

### 7b. Alternative: Restore from Repo Bundle

If you can't copy from old Mac, the key files are committed. But the home-directory files need manual placement:

```bash
# These are the files to recreate in ~/:
# ~/ORCHESTRATOR.md
# ~/agents/implementation-engineer.md
# ~/agents/backend-engineer.md
# ~/agents/frontend-engineer.md
# ~/agents/test-engineer.md
# ~/phase1-plan.md
# ~/phase2-plan.md
```

### 8. Set Up Claude Code Memory

Claude Code memory is stored at a path based on your username and project path. On the new Mac:

```bash
# Figure out your new project path for Claude memory:
# It's based on the CWD where you launch Claude Code from.
# If you clone to ~/dxd-vision-engine, Claude creates:
#   ~/.claude/projects/-Users-YOURUSERNAME/memory/MEMORY.md

# Create the directory:
mkdir -p ~/.claude/projects/-Users-$(whoami)/memory/

# Copy MEMORY.md from old Mac or recreate it.
# The content is in the repo at: .claude/projects/ path
```

### 9. Set Up Git Worktrees

```bash
mkdir -p ~/dxd-worktrees

# Create worktrees for parallel agent development
cd ~/dxd-vision-engine
git worktree add ~/dxd-worktrees/backend backend/phase3
git worktree add ~/dxd-worktrees/frontend frontend/phase3
git worktree add ~/dxd-worktrees/test test/phase3

# Copy Claude settings to each worktree
for wt in backend frontend test; do
  mkdir -p ~/dxd-worktrees/$wt/.claude
  cp ~/dxd-vision-engine/.claude/settings.local.json ~/dxd-worktrees/$wt/.claude/
done
```

### 10. Set Up Tmux Layout

```bash
# Create the 5-pane layout
tmux new-session -s dxd -d

# Split into the standard layout:
# Top-left: Orchestrator, Top-right: Back-End
# Bottom-left: Implementation, Bottom-right-top: Test, Bottom-right-bottom: Front-End
tmux split-window -h -t dxd
tmux split-window -v -t dxd:0.0
tmux split-window -v -t dxd:0.1
tmux split-window -v -t dxd:0.3

# Send cd commands to each pane
tmux send-keys -t dxd:0.0 'cd ~/dxd-vision-engine' Enter        # Orchestrator
tmux send-keys -t dxd:0.1 'cd ~/dxd-vision-engine' Enter        # Implementation
tmux send-keys -t dxd:0.2 'cd ~/dxd-worktrees/backend' Enter    # Back-End
tmux send-keys -t dxd:0.3 'cd ~/dxd-worktrees/test' Enter       # Test
tmux send-keys -t dxd:0.4 'cd ~/dxd-worktrees/frontend' Enter   # Front-End

# Attach
tmux attach -t dxd
```

### 11. Launch Claude Code in Each Pane

In each pane, run `claude` then `/model sonnet` for agents, `/model opus` for orchestrator.

---

## Current Project State (Phase 3)

- **Phase 1:** COMPLETE, tagged `v0.1.0-phase1`
- **Phase 2:** COMPLETE, tagged `v0.2.0-phase2`
- **Phase 3:** Code written, under verification by agents
- **58 tests** passing across all 3 phases
- **9 open beads** remaining (verification tasks)
- Run `export BEADS_NO_DAEMON=1 && bd list` to see current state

## Quick Verify After Setup

```bash
cd ~/dxd-vision-engine
git log --oneline -5                              # Recent commits
git tag -l                                         # Should show v0.1.0-phase1, v0.2.0-phase2
git worktree list                                  # 4 entries
export BEADS_NO_DAEMON=1 && bd list               # Open beads
PYTHONPATH=src python3 -m pytest tests/ -v         # All tests green
```
