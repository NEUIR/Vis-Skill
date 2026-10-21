---
name: alfworld-object-storer
description: Use when the agent is holding an object and needs to place it into a target receptacle in ALFWorld. This skill checks receptacle suitability, opens closed containers if needed, and executes the `put` command to store the object. It handles both open surfaces (countertops, beds) and closed containers (drawers, cabinets).
---
# Skill: Object Storer

## When to Use
Trigger this skill when:
1. The agent is holding the target object (optionally cleaned/heated/cooled as required)
2. A suitable storage receptacle has been identified
3. The agent needs to execute the final placement step

## Core Workflow

### 1. Validate Prerequisites
- Confirm the agent is holding `{object_name}` (check inventory)
- Confirm the agent is at the location of `{target_receptacle}`
- If not at the receptacle, navigate there first: `go to {target_receptacle}`

### 2. Check Receptacle State
Evaluate the receptacle before placing:

| Receptacle State | Action |
|-----------------|--------|
| Open and empty/suitable | Proceed with `put` |
| Closed (drawer, cabinet, safe) | `open {target_receptacle}` first, then `put` |
| Unsuitable (wrong type, full) | Abort and search for alternative receptacle |

### 3. Execute Storage
- Run: `put {object_name} in/on {target_receptacle}`
- Check the observation for confirmation

### 4. Verify Placement
- A successful placement updates the receptacle contents in the observation
- If the observation confirms the object is now in/on the receptacle, storage is complete

## Example

**Task:** "Clean the knife and put it in a drawer."

```
> go to drawer 1
The drawer 1 is closed.
> open drawer 1
You open the drawer 1. The drawer 1 is open. In it, you see nothing.
> put knife 1 in/on drawer 1
You put the knife 1 in/on the drawer 1.
```

**Result:** `knife 1` is now stored in `drawer 1`. Task complete.

**Example 2 — Open surface:**

```
> go to bed 1
On the bed 1, you see a pillow 1.
> put cellphone 2 in/on bed 1
You put the cellphone 2 in/on the bed 1.
```

## Error Handling
- **"Nothing happened"**: The agent may not be holding the object, or the receptacle name is incorrect. Verify with `inventory` and re-check the receptacle identifier.
- **Receptacle unsuitable**: If the receptacle is not appropriate for the object, search for an alternative using the object-locator skill.
- **Agent not at receptacle**: Navigate to the receptacle with `go to {target_receptacle}` before attempting `put`.
