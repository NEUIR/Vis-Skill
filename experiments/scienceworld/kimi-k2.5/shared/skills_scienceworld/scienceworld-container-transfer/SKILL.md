---
name: scienceworld-container-transfer
description: Moves a substance or object from one container to another (e.g., moving lead to a metal pot). Use this skill when you need to prepare materials for processing, such as transferring a substance into a heat-resistant container before heating, or reorganizing materials between vessels for mixing or measurement.
---
# Skill: Container Transfer

## Purpose
Transfer a target substance or object from a source container to a destination container. This is a foundational step for preparing materials for heating, mixing, or measurement.

## Core Workflow
1. **Assess Need:** Determine if the current container is unsuitable for the next operation (e.g., tin cup is not heat-resistant for a furnace).
2. **Locate Items:** Use `look around` and `look at OBJ` to find the target substance and a suitable destination container.
3. **Execute Transfer:** `move <SUBSTANCE> to <DESTINATION>`.
4. **Verify:** `look at <DESTINATION>` to confirm the substance is now inside.

## Key Actions
| Action | Purpose |
|--------|---------|
| `look around` | Survey room for containers and substances |
| `look at OBJ` | Inspect container contents |
| `move OBJ to OBJ` | Transfer substance to destination |

## Example
**Task:** Transfer lead from a tin cup to a heat-resistant metal pot for furnace heating.

1. `look around` — spot `tin cup (containing lead)` and `metal pot`
2. `move lead to metal pot`
3. `look at metal pot` — confirms: "a metal pot (containing a substance called lead)"

## Important Notes
* All containers are pre-opened. Do not use `open` or `close` actions.
* Choose the destination container based on properties needed for the next step (e.g., heat resistance for furnace use).
* If the transfer fails ("You can't do that"), verify object names with `look around` and ensure the destination is accessible.
