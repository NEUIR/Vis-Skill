---
name: scienceworld-tool-validator
description: Use when the agent has acquired a tool or instrument and needs to verify it is operational before first use in a critical task step. This skill performs a lightweight pre-use check via "focus on [TOOL] in inventory" and confirms readiness based on the system's response, ensuring the tool is functional before measurement, activation, or connection operations.
---
# Skill: scienceworld-tool-validator

## Purpose

Perform a pre-use functionality check on a tool or instrument to confirm it is operational before employing it in a critical ScienceWorld task step such as measurement, activation, or connection.

## When to Use

- Immediately after acquiring a tool (e.g., `pick up thermometer`) and before its first use
- When switching to a different tool mid-task and needing to confirm readiness
- When resuming a task after navigation and needing to re-confirm tool availability

## Workflow

1. **Acquire the tool** -- Ensure the target tool is in inventory. If not, locate it with `look around` and retrieve it with `pick up [TOOL]`.
2. **Execute validation** -- Run: `focus on [TOOL] in inventory`.
3. **Confirm readiness** -- A successful response (`"You focus on the [TOOL]."`) confirms the tool is operational. No further diagnostic steps are needed unless an error is observed.
4. **Proceed** -- Use the validated tool in the task operation (e.g., `use thermometer on [TARGET]`).

## Examples

### Example 1: Validating a thermometer before measurement

```
> pick up thermometer
You pick up the thermometer.

> focus on thermometer in inventory
You focus on the thermometer.
```

The thermometer is validated. Proceed with measurement:

```
> use thermometer on unknown substance B
The thermometer measures a temperature of 42 degrees celsius.
```

### Example 2: Validating a scale after teleporting to a new room

```
> teleport to workshop
You teleport to the workshop.

> pick up scale
You pick up the scale.

> focus on scale in inventory
You focus on the scale.
```

The scale is validated and ready to weigh objects.

## Key Principles

- **Timing** -- Validate immediately after acquisition, before any task-sensitive operation.
- **Simplicity** -- The `focus on` action is the primary, lightweight validation method. Avoid unnecessary `examine` or `use` actions during the check.
- **State awareness** -- Ensure containers are open and items are accessible before attempting to pick up tools. Use `teleport to` for efficient navigation.
