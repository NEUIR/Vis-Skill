---
name: alfworld-tool-locator
description: Searches for a specified tool or device (e.g., a desklamp, knife, or sponge) within the ALFWorld environment by checking relevant surfaces. Use when you need a tool to interact with another object as part of a task but the tool is not in your inventory or immediate vicinity. Takes a tool name as implicit input and navigates to likely storage spots (sidetables, shelves, countertops) until the tool is found.
---
# Instructions

Locate a specified tool or device by systematically checking receptacles where it is likely stored.

## Workflow
1. **Prioritize search locations** based on tool type:
   - Small tools/electronics (desklamp, remotecontrol): `sidetable`, `desk`, `drawer`, `shelf`
   - Kitchen tools (knife, spatula): `countertop`, `cabinet`, `drawer`
   - Cleaning tools (sponge, spraybottle): `shelf`, `cabinet`, `countertop`
   - Default: check all visible receptacles in order
2. **Navigate** to each prioritized receptacle: `go to {receptacle}`
3. **Inspect** the observation for the target tool name
4. **If found:** Stop searching. Proceed to use the tool for the main task
5. **If not found:** Continue to the next receptacle in the priority list
6. **If exhausted:** Expand search to remaining receptacles not yet checked

## Action Format
- `go to {receptacle}` -- navigate to check a location
- `take {tool} from {receptacle}` -- pick up the tool once found (if needed)
- `use {tool}` / `toggle {tool}` -- operate the tool in place (e.g., desklamp)

## Error Recovery
- If "Nothing happened": verify the receptacle name matches the environment scan
- If tool not found after all likely spots: expand search to every receptacle from the initial environment scan

## Example

**Scenario:** You need to find a desklamp to examine an object.

```
Thought: I need to find a desklamp. Desklamps are usually on sidetables or desks.
Action: go to sidetable 1
Observation: On the sidetable 1, you see a keychain 1.
Thought: No desklamp here. Let me check the next sidetable.
Action: go to sidetable 2
Observation: On the sidetable 2, you see a desklamp 1.
```

**Result:** Found `desklamp 1` on `sidetable 2`. Proceed to use it for the task.
