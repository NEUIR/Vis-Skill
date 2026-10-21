---
name: alfworld-appliance-navigator
description: Navigates the agent to a target appliance (microwave, stove, fridge, or sinkbasin) needed for object processing. Use when you are holding an object that needs heating, cooling, or cleaning and must move to the correct appliance station. Identifies the required appliance from the task context and executes the movement action.
---
# Instructions
Use this skill when you are holding an object that needs to be processed (heated, cooled, or cleaned) and you must locate and move to the correct appliance to perform the action.

## Workflow
1. **Identify the Target Appliance:** Determine which appliance is required for the task. Map the action to the appliance: `heat` -> microwave/stoveburner, `cool` -> fridge, `clean` -> sinkbasin.
2. **Locate the Appliance:** Scan the environment observation for the target appliance (e.g., `microwave 1`, `fridge 1`, `sinkbasin 1`).
3. **Navigate:** Execute `go to {appliance}` to move to the identified appliance location.
4. **Prepare Appliance (if needed):** Upon arrival, check if the appliance requires preparation (e.g., opening a closed microwave or fridge door). If so, perform `open {appliance}` before proceeding.

## Example

**Scenario:** You are holding `potato 1` and need to heat it.

```
Thought: I need to heat this potato. The microwave is the appropriate appliance.
Action: go to microwave 1
Observation: The microwave 1 is closed.
Thought: I need to open the microwave before I can use it.
Action: open microwave 1
Observation: You open the microwave 1. The microwave 1 is open. In it, you see nothing.
```

**Result:** You are now at the open microwave, ready to heat the potato.

## Key Principles
- **Trigger:** The agent is holding an object and the next step in the task is to `heat`, `cool`, or `clean` it.
- **Core Action:** The primary output of this skill is the navigation command `go to {target_appliance}`.
- **Prerequisite Check:** Always ensure the appliance is accessible (e.g., open) before attempting to use it for processing.
