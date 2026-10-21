---
name: scienceworld-conditional-focus-executor
description: Executes a 'focus on OBJ' action on a specific object based on the outcome of a prior conditional evaluation. Use when you have a measurement result and task instructions specify focusing on different objects (e.g., colored boxes) depending on whether the result meets a threshold.
---
# Conditional Focus Executor

## When to Use
Use after completing a measurement (temperature, pH, mass) when the task specifies a conditional rule like "If result > X, focus on A; otherwise, focus on B."

## Procedure
1. **Evaluate the condition** against your measurement result.
2. **Determine the target object** based on which branch is true.
3. **Execute:** `focus on <TARGET_OBJECT>`
4. **Verify:** Confirm the focus action succeeded before proceeding.

If the focus action fails (e.g., object not found), use `look around` to verify the target object name matches exactly.

## Example
**Task:** "If the temperature of the substance is above 50C, focus on the red box. Otherwise, focus on the blue box."
1. Measurement result: 63C
2. Evaluation: 63 > 50 is true → target is red box
3. Execute: `focus on red box`
4. Observation: "You focus on the red box." → task complete.
