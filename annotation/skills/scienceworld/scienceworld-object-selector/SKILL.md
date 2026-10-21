---
name: scienceworld-object-selector
description: Use when the agent needs to choose a specific object from multiple candidates in the environment based on task criteria such as object type (non-living thing, electrical component, container), properties, or category. This skill surveys visible objects with "look around", evaluates each against the task requirements, selects the most suitable candidate, and signals intent with "focus on [OBJECT]".
---
# Skill: scienceworld-object-selector

## Purpose

Identify and select the correct object from visible candidates in the ScienceWorld environment based on task-defined criteria (e.g., "non-living thing", "electrical component", "container"), then signal intent with a `focus on` action.

## When to Use

- When a task requires choosing a specific type of object from multiple visible options
- When the agent must classify objects by property (living/non-living, material type, function) to select one
- When the task instructs finding and interacting with a particular category of object

## Workflow

1. **Survey the environment** -- Use `look around` to list all visible objects in the current location.
2. **Evaluate candidates** -- For each visible object, determine if it matches the task criteria. Exclude target containers (e.g., destination boxes) and fixed furniture from candidates.
3. **Select the best match** -- Choose the first suitable, portable object unless the task specifies a preference (e.g., "largest", "closest").
4. **Signal intent** -- Execute `focus on [OBJECT]` to confirm selection.
5. **Execute the task action** -- Proceed with the required action (e.g., `pick up`, `move`, `use`).

## Examples

### Example 1: Find a non-living thing and move it

**Task:** "Find a non-living thing. First, focus on the thing. Then, move it to the purple box."

```
> look around
You see: purple box, table, battery, black wire, blue light bulb, red light bulb, red wire, switch, violet light bulb, yellow wire, ultra low temperature freezer.
```

**Evaluation:** Purple box is the destination. Table and freezer are fixed furniture. Remaining items (battery, wires, bulbs, switch) are all non-living and portable.

```
> focus on black wire
You focus on the black wire.

> move black wire to purple box
You move the black wire to the purple box.
```

### Example 2: Find an electrical component

**Task:** "Find an electrical component and place it on the workbench."

```
> look around
You see: workbench, chair, battery, glass cup, red wire, notebook.

> focus on battery
You focus on the battery.

> move battery to workbench
You move the battery to the workbench.
```

## Key Principles

- **Exclude destinations** -- Never select the target container or destination as the candidate object.
- **Verify if uncertain** -- Use `examine [OBJECT]` for more detail about an object's properties before selecting.
- **Task alignment** -- The selection should enable the next action in the task sequence.
