---
name: scienceworld-task-focuser
description: Use when the agent needs to direct attention to a specific object in the environment or inventory before performing a critical action such as measuring, using, or connecting. This preparatory "focus on" step signals intent to the ScienceWorld system, often unlocking required state changes for subsequent interactive steps in experimental procedures.
---
# Skill: scienceworld-task-focuser

## Purpose

Formally declare intent to interact with a specific object in the ScienceWorld environment. The `focus on` action is a required preparatory step that signals to the system which object subsequent actions (`use`, `measure`, `connect`) will target, often unlocking necessary state changes or measurement capabilities.

## When to Use

- Before using a measurement tool (e.g., thermometer) on a target object
- Before complex operations such as connecting, mixing, or combining objects
- When a task description explicitly instructs "focus on" an object
- To confirm the correct item is in inventory before proceeding with an action

## Workflow

1. **Identify the target object** -- Determine the exact name as it appears in the environment (e.g., `thermometer`, `metal fork`).
2. **Locate the object** -- Use `look around` or `inventory` to confirm the object's location (environment or inventory).
3. **Execute the focus action** -- Issue the appropriate command:
   - Object in the environment: `focus on <OBJECT_NAME>`
   - Object in inventory: `focus on <OBJECT_NAME> in inventory`
4. **Verify confirmation** -- Wait for the system response: `"You focus on the <OBJECT_NAME>."` before proceeding to the next step.

## Examples

### Example 1: Preparing to measure temperature of a metal fork

```
> focus on thermometer in inventory
You focus on the thermometer.

> focus on metal fork in inventory
You focus on the metal fork.

> use thermometer on metal fork
The thermometer measures a temperature of 23 degrees celsius.
```

### Example 2: Focusing on an environmental object before moving it

```
> look around
You see: a table, a blue box, a red box, a metal fork (on table).

> focus on metal fork
You focus on the metal fork.

> move metal fork to blue box
You move the metal fork to the blue box.
```

## Important Notes

- This skill does **not** perform the main action (e.g., measuring). It only prepares for it.
- The `focus` action has no physical effect on the object; it is a declarative meta-action.
- Always wait for the confirmation observation before attempting the next step.
- If the object is not found, use `look around`, `examine`, or `teleport to` to locate it before attempting to focus.
