---
name: scienceworld-measurement-taker
description: Use when the agent needs to measure a quantitative property (temperature, weight, pH) of a target object or substance using a measurement tool. This skill covers acquiring the tool, preparing both tool and target with focus actions, executing the measurement via "use [TOOL] on [TARGET]", and interpreting the resulting value for conditional decisions such as classification or placement.
---
# Skill: scienceworld-measurement-taker

## Purpose

Measure a quantitative property of a target object or substance in the ScienceWorld environment using the appropriate measurement tool, then interpret the result for subsequent decision-making.

## When to Use

- When a task requires obtaining a numerical reading (temperature, weight, pH) from an object or substance
- When a measured value determines a conditional next action (e.g., classify based on temperature threshold)
- When the agent needs to compare properties across multiple objects

## Workflow

1. **Identify and acquire the tool** -- Locate the correct measurement instrument (e.g., thermometer, scale) and `pick up` the tool.
2. **Focus on the tool** -- Execute `focus on [TOOL] in inventory` to confirm readiness.
3. **Identify and acquire the target** -- Locate the target object or substance and `pick up` the target.
4. **Focus on the target** -- Execute `focus on [TARGET] in inventory` to confirm readiness.
5. **Position for follow-up** -- If the task requires a follow-up action (e.g., placing in a bin), `teleport to` the appropriate location before measuring.
6. **Execute measurement** -- Use the tool on the target: `use [TOOL] on [TARGET]`.
7. **Interpret and act** -- Read the numerical result and execute the appropriate conditional action.

## Examples

### Example 1: Measure temperature and classify

```
> teleport to kitchen
You teleport to the kitchen.

> pick up thermometer
You pick up the thermometer.

> focus on thermometer in inventory
You focus on the thermometer.

> pick up unknown substance B
You pick up the unknown substance B.

> focus on unknown substance B in inventory
You focus on the unknown substance B.

> teleport to hallway
You teleport to the hallway.

> use thermometer on unknown substance B
The thermometer measures a temperature of 56 degrees celsius.

> move unknown substance B to orange box
You move the unknown substance B to the orange box.
```

### Example 2: Measure and compare two objects

```
> use thermometer on metal fork
The thermometer measures a temperature of 23 degrees celsius.

> use thermometer on glass cup
The thermometer measures a temperature of 87 degrees celsius.
```

The glass cup (87) is hotter than the metal fork (23).

## Key Principles

- **Tool first** -- Secure the measurement tool before handling the target.
- **Focus before use** -- Always `focus on` both tool and target in inventory before measuring.
- **Plan ahead** -- Identify where follow-up actions occur and position accordingly before measuring.
- **Map thresholds** -- Clarify decision thresholds (e.g., "above 100.0" vs. "below 100.0") before executing the measurement.
