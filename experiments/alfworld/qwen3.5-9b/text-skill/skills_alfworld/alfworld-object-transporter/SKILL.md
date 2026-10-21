---
name: alfworld-object-transporter
description: Picks up a target object from its current receptacle and moves it to a specified destination receptacle. Use when you have located an object and need to relocate it to complete a task (e.g., moving a laptop to a desk). Takes the object identifier, source receptacle, and destination receptacle as inputs and outputs the action sequence to take, transport, and place the object.
---
# Instructions

Pick up an object from its current location and transport it to a destination receptacle.

## Workflow
1. **Navigate to source:** `go to {source_receptacle}` -- verify observation shows the target object
2. **Pick up:** `take {object} from {source_receptacle}` -- verify "You pick up" confirmation
3. **Navigate to destination:** `go to {destination_receptacle}`
4. **Place:** `put {object} in/on {destination_receptacle}` -- verify "You put" confirmation

## Action Format
- `go to {receptacle}`
- `take {object} from {receptacle}`
- `put {object} in/on {receptacle}`

## Error Recovery
- "Nothing happened" on take: verify you are at the correct receptacle and the object name matches the observation
- "Nothing happened" on put: verify you are holding the object and at the correct destination
- Object not visible: re-scan the environment to locate it before retrying

## Example

**Scenario:** Move `laptop 1` from `bed 2` to `desk 1`.

```
Action: go to bed 2
Observation: On the bed 2, you see a laptop 1, a pillow 1.
Action: take laptop 1 from bed 2
Observation: You pick up the laptop 1 from the bed 2.
Action: go to desk 1
Observation: On the desk 1, you see a pen 2.
Action: put laptop 1 in/on desk 1
Observation: You put the laptop 1 in/on the desk 1.
```

**Result:** The laptop has been transported from the bed to the desk.

## Bundled Resources
- **Script**: `scripts/transport_sequence.py` provides a deterministic sequence generator.
- **Reference**: `references/troubleshooting.md` contains common failure patterns and solutions.
