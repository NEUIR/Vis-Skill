---
name: alfworld-object-state-inspector
description: Inspects a receptacle's contents by navigating to it and reading the observation. Use when you need to check what is on or inside a receptacle (e.g., "what's on the shelf", "is the holder empty", "check the table for items"). Executes `go to {receptacle}`, parses the observation listing items present, and decides whether to take an item, search elsewhere, or proceed.
---
# Instructions

Inspect the state or contents of a target receptacle by navigating to it and parsing the environment's observation feedback.

## Workflow
1. **Navigate:** Execute `go to {target_receptacle}`
2. **Read observation:** The environment automatically reports what is on/in the receptacle -- no additional inspection action is needed
3. **Parse contents:** Look for patterns:
   - `"On the {receptacle}, you see nothing."` -- receptacle is empty
   - `"On the {receptacle}, you see a {item1}, and a {item2}."` -- items are present
4. **Decide next action** based on the observation:
   - Empty: search elsewhere for the needed item
   - Item found: `take {item} from {receptacle}`
   - Wrong items: move on to the next receptacle

## Error Recovery
- "Nothing happened": the `go to` target name is invalid -- verify the receptacle name from your environment scan
- This skill uses only `go to` for navigation; it does not use `open`, `close`, or `toggle`

## Example

**Scenario 1:** Check if a toiletpaperhanger has toilet paper.

```
Action: go to toiletpaperhanger 1
Observation: On the toiletpaperhanger 1, you see nothing.
```

**Decision:** Holder is empty. Find a toiletpaper roll elsewhere and bring it here.

**Scenario 2:** Check a toilet for available items.

```
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.
```

**Decision:** toiletpaper 1 is available. Execute `take toiletpaper 1 from toilet 1`.
