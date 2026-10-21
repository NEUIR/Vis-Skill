---
name: alfworld-clean-object
description: Cleans a specified object using an appropriate cleaning receptacle (e.g., sinkbasin). Use when a task requires an object to be in a clean state (e.g., "clean potato", "wash apple") before proceeding. Navigates to the cleaning location, performs the clean action, and confirms the object is now clean.
---
# Instructions

Clean an object you are holding using a sinkbasin. The object must be in your inventory before cleaning.

## Workflow
1. **Navigate:** `go to sinkbasin 1` (or the appropriate sinkbasin in the environment)
2. **Clean:** `clean {object} with sinkbasin 1` -- verify observation confirms "You clean the {object}"
3. **Proceed:** The object is now clean. Continue with the next task step

## Action Format
- `clean {obj} with {recep}` (e.g., `clean potato 1 with sinkbasin 1`)

## Error Recovery
- "Nothing happened": Check (1) you are holding the object, (2) you are at the sinkbasin, (3) object and receptacle names are correct
- Not at sinkbasin: execute `go to sinkbasin 1` first

## Example

**Scenario:** You are holding `potato 1` and need to clean it.

```
Thought: I need to clean this potato. I should go to the sinkbasin.
Action: go to sinkbasin 1
Observation: On the sinkbasin 1, you see nothing.
Action: clean potato 1 with sinkbasin 1
Observation: You clean the potato 1 using the sinkbasin 1.
```

**Result:** The potato is now in a clean state and ready for the next task step.

## Post-Condition
After successful execution, the object will be in a clean state. You may proceed with the next step of your task (e.g., placing the clean object on a shelf or in a microwave).
