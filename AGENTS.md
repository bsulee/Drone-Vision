# Agent Instructions for DXD Vision Engine

## Workflow
1. Check work: `bd ready`
2. Claim work: `bd update <id> --status in_progress`
3. While working: Update frequently with `bd update <id> --comment "status"`
4. When done: `bd close <id> --reason "what you did"`
5. End session: "Land the plane" (see below)

## Landing the Plane
At session end:
1. Commit/stash all changes
2. Update all beads with current status
3. File new beads for discovered work
4. Run `bd ready` and generate next session prompt
5. Output: "Next session: Continue bd-XYZ - [context]"

## Rules
- Never work without a bead
- Keep descriptions actionable
- Update beads often
- Check for blockers
