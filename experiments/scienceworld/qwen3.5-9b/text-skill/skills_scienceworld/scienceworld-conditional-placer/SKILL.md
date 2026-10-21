---
name: scienceworld-conditional-placer
description: Places an object into one of several designated containers based on a measured condition, such as a temperature threshold. Use this skill when you have completed a measurement or assessment and the task requires sorting or storing the object into one of multiple containers according to a rule (e.g., "if temperature > X, place in container A; otherwise container B").
---
# Skill: Conditional Object Placer

## Purpose
Place a target object into the correct container based on a measured condition (e.g., temperature threshold, conductivity result). This skill executes the full measure-then-sort workflow.

## Core Workflow
1. **Locate & Acquire Measurement Tool:** `teleport to LOC` then `pick up` the measurement device (e.g., thermometer).
2. **Locate & Acquire Target Object:** `teleport to LOC` then `pick up` the object to be measured (e.g., metal fork).
3. **Identify Target Containers:** Use `look around` to find the designated containers (e.g., blue box, orange box).
4. **Perform Measurement:** `use OBJ on OBJ` (e.g., `use thermometer on metal fork`) to obtain the value.
5. **Evaluate Condition & Place:** Compare the measured value against the threshold, then `move OBJ to OBJ` to place the object in the correct container.

## Key Actions
| Action | Purpose |
|--------|---------|
| `teleport to LOC` | Navigate between rooms |
| `look around` | Survey a room for objects |
| `pick up OBJ` | Acquire tools or target object |
| `use OBJ on OBJ` | Perform measurement |
| `move OBJ to OBJ` | Place object into selected container |

## Example
**Task:** "Measure the temperature of the metal fork. If above 50C, place in the orange box. Otherwise, place in the blue box."

1. `teleport to kitchen`
2. `look around` — find thermometer and metal fork
3. `pick up thermometer`
4. `pick up metal fork`
5. `use thermometer on metal fork` — reads 72 degrees
6. 72 > 50, so: `move metal fork to orange box`

## Important Notes
* All containers are pre-opened. Do not use `open` or `close` actions.
* The measurement tool and target object must be picked up and used from inventory.
* Room names, object names, thresholds, and container names vary per task — adapt accordingly.
