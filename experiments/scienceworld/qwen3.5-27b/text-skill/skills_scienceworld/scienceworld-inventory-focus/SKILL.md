---
name: scienceworld-inventory-focus
description: Use when the agent needs to confirm and prepare a specific inventory item before using it in an experiment or task step. This "focus on [ITEM] in inventory" action verifies the correct item has been collected and signals intent, ensuring operational readiness for subsequent actions like measurement, combination, or placement.
---
# Skill: scienceworld-inventory-focus

## Purpose

Confirm and prepare an inventory item before using it in a ScienceWorld task. The `focus on [ITEM] in inventory` action verifies the item's presence and signals intent, reducing errors in multi-step experimental procedures.

## When to Use

- After picking up an item that will be used in a subsequent critical action (measurement, combination, connection)
- When verifying the correct object has been collected before proceeding
- Before using a tool on a target object, to prepare both tool and target

## Workflow

1. **Acquire the item** -- Ensure the target item is in inventory (e.g., `pick up thermometer`).
2. **Focus on the item** -- Execute: `focus on [ITEM] in inventory` (replace `[ITEM]` with the exact object name).
3. **Confirm readiness** -- Wait for system response: `"You focus on the [ITEM]."`.
4. **Proceed** -- Execute the intended action (e.g., `use thermometer on unknown substance B`).

## Examples

### Example 1: Preparing tool and target for temperature measurement

```
> pick up thermometer
You pick up the thermometer.

> focus on thermometer in inventory
You focus on the thermometer.

> pick up unknown substance B
You pick up the unknown substance B.

> focus on unknown substance B in inventory
You focus on the unknown substance B.

> use thermometer on unknown substance B
The thermometer measures a temperature of 42 degrees celsius.
```

### Example 2: Preparing an item for placement

```
> pick up metal fork
You pick up the metal fork.

> focus on metal fork in inventory
You focus on the metal fork.

> move metal fork to blue box
You move the metal fork to the blue box.
```

## Important Notes

- The `focus on` action does not change the object's state; it is a declarative checkpoint that changes the agent's awareness and intent.
- Always use the exact object name as it appears in the inventory.
- If the item is not in inventory, use `pick up [ITEM]` first before attempting to focus.
