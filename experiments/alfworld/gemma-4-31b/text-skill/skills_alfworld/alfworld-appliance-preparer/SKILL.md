---
name: alfworld-appliance-preparer
description: Prepares a household appliance (microwave, oven, toaster, fridge) for use by ensuring it is in the correct open/closed state. Use when the agent needs to heat, cool, or cook an item and must first open or close the appliance before placing an object inside. Takes an appliance identifier as input and outputs a confirmation that the appliance is ready for the next action.
---
# Instructions

## Goal
Prepare a specified household appliance for immediate use by ensuring it is in the correct open or closed state. This is a prerequisite step before performing actions like `heat`, `cool`, or `toggle` with the appliance.

## Input
- **appliance_identifier**: The name of the appliance to prepare (e.g., `microwave 1`, `toaster 1`, `fridge 1`).

## Core Logic
1.  **Navigate to the Appliance**: First, go to the location of the target appliance.
2.  **Check State & Prepare**: Determine if the appliance needs to be opened or closed based on the intended subsequent action. The standard rule is:
    *   For appliances used to contain items for processing (e.g., microwave, oven, fridge), they typically need to be **open** to receive the item.
    *   Use the `open {appliance}` or `close {appliance}` action as needed.
3.  **Confirm Readiness**: The skill is complete when the appliance is in the correct state, confirmed by an observation from the environment (e.g., "The microwave 1 is open.").

## Important Considerations
- **State Awareness**: Always observe the environment's feedback after each action (e.g., "The microwave 1 is closed."). Do not assume the state.
- **Error Handling**: If the action fails (environment outputs "Nothing happened"), the appliance may already be in the desired state. Re-check the observation and proceed.
- **Trajectory Insight**: Refer to the example in `references/trajectory_example.md` to see a practical application of this skill in the context of a larger task.

## Example

**Input:** `appliance_identifier: microwave 1`

**Sequence:**
1. `go to microwave 1` → Observation: "You are at microwave 1. The microwave 1 is closed."
2. `open microwave 1` → Observation: "You open the microwave 1. The microwave 1 is open."

**Output:** "The microwave 1 is open and ready for use."

## Output
A confirmation that the appliance is ready, typically in the form of the agent's `Thought` summarizing the prepared state and the environment's observation.
