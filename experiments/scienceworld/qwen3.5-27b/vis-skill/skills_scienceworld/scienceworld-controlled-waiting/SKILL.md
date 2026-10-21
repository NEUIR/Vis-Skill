---
name: controlled-waiting
description: Executes timed waiting using 'wait' or 'wait1' actions to advance the simulation clock. Use when a time-dependent process like plant growth, chemical reaction, or mechanical cycle must progress before you can continue the task.
---
# Skill: Controlled Waiting

## When to Use
After initiating a process with a time delay, when no other productive actions are possible.

## Procedure
1. Confirm the process is active (seeds planted, device activated, etc.).
2. `wait` (advances 10 steps) for long processes, or `wait1` (1 step) for fine-grained observation.
3. `look around` or `examine <OBJECT>` to check if the target state has been reached.
4. If not reached, repeat steps 2-3. If reached, exit and resume the main task.

**Duration guidance:** Use `wait` for biological growth stages. Use `wait1` when observing rapid changes or when close to the expected transition.

## Example
**Task:** Wait for a banana tree to produce fruit after planting.
1. Confirm seeds are planted and watered.
2. `wait` (advances 10 steps)
3. `look around` — check if banana is visible on the tree.
4. No banana yet → `wait` again.
5. `look around` — observe: "a banana tree (with a banana)" → exit skill and proceed to harvest.

Refer to `references/botanical_growth_patterns.md` for common plant growth stage timings.
