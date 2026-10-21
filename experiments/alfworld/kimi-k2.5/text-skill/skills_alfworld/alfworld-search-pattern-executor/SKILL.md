---
name: alfworld-search-pattern-executor
description: Systematically searches a sequence of likely locations for a target object based on common sense. Use when you need to find a specific object and know which receptacles to check but not which one contains it. Takes a list of candidate receptacles, orchestrates navigation and inspection, and outputs when the target is found or all locations are exhausted.
---
# Instructions

For each candidate receptacle in order, execute this action sequence:

## Workflow
1. **Navigate:** `go to {receptacle}`
2. **Open if closed:** If observation says "{receptacle} is closed", execute `open {receptacle}`
3. **Inspect:** Read the observation for the target object name
4. **If found:** Execute `take {target_object} from {receptacle}` and stop searching
5. **If not found:** If you opened the receptacle, execute `close {receptacle}`, then continue to the next candidate
6. **If all exhausted:** Report that the target was not found in any candidate location

## Action Format
- `go to {receptacle}` -- navigate to a location
- `open {receptacle}` -- open a closed drawer/cabinet/fridge
- `close {receptacle}` -- close after checking (state management)
- `take {object} from {receptacle}` -- pick up the target when found

## Key Rules
- Search candidates in order; do not skip locations
- Always close drawers/cabinets after checking if you opened them
- Once the object is found, immediately stop searching and proceed to the next task phase
- If "Nothing happened" is returned, verify the receptacle name and retry with the correct identifier

## Example

**Goal:** "find two remotecontrol and put them in armchair."

```
Thought: I need to find remotecontrol. Candidates: sofa 1, sidetable 1, coffeetable 1. Starting search.
Action: go to sofa 1
Observation: On the sofa 1, you see a pillow 1.
Thought: No remotecontrol on sofa 1. Checking next candidate.
Action: go to sidetable 1
Observation: On the sidetable 1, you see a keychain 1.
Thought: Not here either. Checking coffeetable.
Action: go to coffeetable 1
Observation: On the coffeetable 1, you see a remotecontrol 1, a remotecontrol 2.
Action: take remotecontrol 1 from coffeetable 1
Observation: You pick up the remotecontrol 1 from the coffeetable 1.
```

**Result:** First remotecontrol found and picked up. Proceed to place it in `armchair 1`, then re-activate search for the second one.
