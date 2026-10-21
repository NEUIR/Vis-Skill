---
name: task-completion-focus
description: Focuses on a specific target object to signal task completion. Use when you have produced the required final object (like a grown banana) and need to formally complete the assigned task. This handles the 'focus on OBJ' action that typically marks successful task execution in the environment.
---
# Skill: Task Completion Focus

## When to Use
The final required object (e.g., grown banana, crafted item) is visibly present and you need to formally signal task completion. Do not use for intermediate steps.

## Procedure
1. `look around` — confirm the target object is visible in the scene.
2. `focus on <OBJECT>` — signal task completion.
3. **Ambiguity handling:** If the environment returns "Ambiguous request" with numbered options, respond with the option number (e.g., `0`) for the target most directly associated with your task goal.

## Example
**Task:** "Grow a banana." After successful cultivation:
1. `look around` — observation: "On the banana tree you see: a banana, a flower."
2. `focus on banana on banana tree`
3. If ambiguous prompt appears listing multiple bananas, select `0` (first instance).
4. Observation confirms task completion.
