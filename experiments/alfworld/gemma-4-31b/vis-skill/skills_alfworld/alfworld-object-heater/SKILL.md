---
name: alfworld-object-heater
description: Heats a specified object using an available heating appliance (e.g., microwave, stoveburner). Use when you are holding an object that requires heating and need to navigate to and operate the heating appliance. Takes the object and appliance as inputs and results in the object being in a heated state.
---
# Instructions

Heat an object you are holding using a compatible heating appliance (microwave or stoveburner).

## Prerequisites
- The target object must be in your inventory
- A heating appliance (microwave, stoveburner) must exist in the environment

## Workflow
1. **Navigate:** `go to {appliance}` (e.g., `go to microwave 1`)
2. **Check state:** If observation says appliance is closed, execute `open {appliance}` -- verify observation confirms it is now open
3. **Heat:** `heat {object} with {appliance}` -- verify observation confirms "You heat the {object}"
4. **(Optional) Place:** If the task requires it, navigate to the destination and place the heated object

## Action Format
- `go to microwave 1` / `go to stoveburner 1`
- `open microwave 1`
- `heat {object} with {appliance}` (e.g., `heat potato 1 with microwave 1`)

## Error Recovery
- "Nothing happened" on heat: Check (1) you are holding the object, (2) appliance is open, (3) appliance name is correct
- Appliance occupied: Take the existing item out first, then retry

## Example

**Scenario:** You are holding `potato 1` and need to heat it.

```
Thought: I need to heat the potato. I'll go to the microwave.
Action: go to microwave 1
Observation: The microwave 1 is closed.
Action: open microwave 1
Observation: You open the microwave 1. The microwave 1 is open. In it, you see nothing.
Action: heat potato 1 with microwave 1
Observation: You heat the potato 1 using the microwave 1.
```

**Result:** The potato is now heated. Proceed to place it at the destination if required by the task.
