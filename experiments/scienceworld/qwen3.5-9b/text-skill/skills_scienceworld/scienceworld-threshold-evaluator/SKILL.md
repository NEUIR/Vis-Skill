---
name: scienceworld-threshold-evaluator
description: Use when the agent has just obtained a numerical measurement (temperature, weight, pH) and must compare it against a predefined threshold to determine a binary outcome. This skill extracts the measured value, evaluates it against the threshold condition (above/below), and executes the corresponding branch action such as classification or placement.
---
# Skill: scienceworld-threshold-evaluator

## Purpose

Compare a measured numerical value against a predefined threshold to determine which of two conditional actions to execute. This is the decision-making step that immediately follows a measurement in ScienceWorld experimental workflows.

## When to Use

- Immediately after obtaining a numerical measurement (e.g., temperature reading from a thermometer)
- When the task includes a conditional instruction like "if above X, do A; if below X, do B"
- When classifying or sorting objects based on measured properties

## Workflow

1. **Extract the measurement** -- Parse the numerical value from the observation (e.g., `"the thermometer measures a temperature of 56 degrees celsius"` yields `56`).
2. **Identify the threshold and condition** -- From the task instruction, determine the threshold value and comparison operator (e.g., `"above 50.0 degrees"` means `threshold=50.0`, `operator=">"`).
3. **Evaluate the comparison** -- Compare: `measured_value > threshold` or `measured_value < threshold`.
4. **Execute the correct branch** -- Perform the action specified for the satisfied condition.

## Examples

### Example 1: Temperature-based classification

**Task:** "Measure the temperature. If above 50.0 degrees, move to the orange box. If below 50.0 degrees, move to the blue box."

```
> use thermometer on unknown substance B
The thermometer measures a temperature of 56 degrees celsius.
```

**Evaluation:** 56 > 50.0 is TRUE, so execute the "above" branch.

```
> move unknown substance B to orange box
You move the unknown substance B to the orange box.
```

### Example 2: Weight-based sorting

**Task:** "If the object weighs more than 200 grams, place in the red bin. Otherwise, place in the green bin."

```
> use scale on rock sample
The scale measures a weight of 145 grams.
```

**Evaluation:** 145 > 200 is FALSE, so execute the "otherwise" branch.

```
> move rock sample to green bin
You move the rock sample to the green bin.
```

## Key Principles

- **Immediate execution** -- Do not perform other actions between obtaining the measurement and evaluating the threshold.
- **Precision** -- Use the exact numerical value from the observation; do not estimate or round.
- **Binary decision** -- The outcome is strictly one of two paths. If the measurement equals the threshold, re-examine the instruction for boundary guidance ("above" typically means `>`, not `>=`).

## Common Pitfalls

- **Incorrect branching** -- Executing the action for the opposite condition (e.g., blue box when value is above threshold).
- **Premature evaluation** -- Attempting to evaluate before the measurement is complete and valid.
- **Action confusion** -- Targeting the wrong object in the post-evaluation action.
