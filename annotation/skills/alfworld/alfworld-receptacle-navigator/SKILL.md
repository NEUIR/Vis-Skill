---
name: alfworld-receptacle-navigator
description: Plans and executes movement to a target receptacle within the ALFWorld environment. Use when the agent needs to reach a specific location before interacting with objects there (e.g., go to fridge to cool an item, go to garbagecan to dispose of an item). Takes a receptacle identifier as input, executes the `go to` action, and outputs confirmation of arrival at the destination.
---
# Skill: Receptacle Navigator

## Purpose
Navigate to a specified receptacle (e.g., `fridge 1`, `garbagecan 1`, `diningtable 1`) in an ALFWorld household environment. This is a foundational skill for positioning the agent to perform subsequent object interactions.

## When to Use
Trigger this skill when the agent's plan requires:
1. **Accessing a location** to interact with an object there (e.g., "go to fridge to cool an item").
2. **Interacting with the receptacle itself** (e.g., "go to garbagecan to dispose of an item").

## Core Logic
1.  **Input Parsing:** Extract the target receptacle name (e.g., `fridge 1`) from the current task context or observation.
2.  **Action Execution:** Output the ALFWorld action: `go to {target_receptacle}`.
3.  **Success Verification:** Confirm the action was successful by checking the observation for evidence of arrival (e.g., "You are at the {receptacle}" or the receptacle is now in the observable list).

## Instructions
1.  Identify the target receptacle from the goal or current observation.
2.  Execute the `go to` action.
3.  If the environment responds with "Nothing happened", the action was invalid. Consult the `references/common_receptacles.md` list and the latest observation to formulate a new `go to` target.
4.  Upon successful navigation, proceed with the next skill in the plan (e.g., `alfworld-object-interactor`).

## Example

**Input:** `target_receptacle: fridge 1`

**Sequence:**
1. `go to fridge 1` → Observation: "You are at fridge 1. On the fridge 1, you see nothing."

**Output:** Agent has arrived at fridge 1 and can now interact with it (open, take items, cool objects, etc.).

## Error Handling
- **Invalid Target:** If "Nothing happened" is observed, the receptacle may be unreachable or incorrectly named. Re-analyze the scene description to find a valid path or alternative receptacle.
- **Already at Location:** If the observation indicates you are already at the target, skip the `go to` action and proceed.
