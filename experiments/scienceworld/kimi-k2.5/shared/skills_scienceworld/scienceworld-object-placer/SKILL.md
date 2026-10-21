---
name: scienceworld-object-placer
description: Moves a specified object from the environment or inventory into a target container based on a classification decision. Use this skill when a task requires sorting or storing an object in a specific location after an assessment (e.g., placing a conductive object in the blue box or a non-conductive object in the orange box).
---
# Skill: Object Placer

## Purpose
Move an object into the correct container based on a prior classification or assessment result. This is the final step in conditional sorting workflows.

## Core Workflow
1. **Confirm Assessment Result:** Know which container corresponds to which classification (e.g., "blue box" = conductive, "orange box" = non-conductive).
2. **Acquire Object (if needed):** `pick up OBJ` if the target is not already in inventory.
3. **Execute Placement:** `move OBJ to OBJ` — place the object in the correct container.
4. **Verify Placement:** `look at <CONTAINER>` to confirm the object is now inside.

## Key Actions
| Action | Purpose |
|--------|---------|
| `pick up OBJ` | Acquire object if not in inventory |
| `move OBJ to OBJ` | Place object into destination container |
| `look at OBJ` | Verify placement succeeded |

## Example
**Task:** "Determine if metal pot is electrically conductive. If conductive, place in the blue box. If nonconductive, place in the orange box."

1. Assessment complete: circuit test confirmed metal pot is conductive.
2. `pick up metal pot` (if not already held)
3. `move metal pot to blue box`
4. `look at blue box` — confirms: "In the blue box is: a metal pot"

## Important Notes
* The classification or assessment must be completed before invoking this skill.
* Always verify placement with `look at` — do not assume success.
* Container names vary per task (blue box, orange box, etc.) — match them to the task instructions.
