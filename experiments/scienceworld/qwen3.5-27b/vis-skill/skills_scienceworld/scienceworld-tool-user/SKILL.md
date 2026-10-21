---
name: scienceworld-tool-user
description: Uses a tool from inventory on a target object or location to perform a specific environmental interaction, such as digging, cutting, or measuring. Use this skill when a task requires modifying the environment or manipulating materials with a tool (e.g., using a shovel to dig soil, a thermometer to measure temperature, or an axe to cut wood). Takes the tool and target as inputs and outputs the result of the interaction.
---
# Skill: Tool User

## Purpose
Perform a physical action on the environment using a tool from your inventory. The core action is `use TOOL on TARGET`.

## When to Use
- A task requires modifying the environment (digging, cutting, heating).
- You need to measure a property of an object (temperature, conductivity).
- A tool-based interaction is the next step in your experiment workflow.

## Core Workflow
1. **Ensure Tool in Inventory:** The required tool (e.g., `shovel`, `thermometer`, `axe`) must be in your inventory. Use `pick up OBJ` if it is not.
2. **Identify Target:** Determine the exact target object or location name (e.g., `ground`, `tree`, `metal fork`).
3. **Execute:** `use TOOL on TARGET`.
4. **Interpret Result:** Read the observation output to determine the outcome (e.g., temperature reading, state change).

## Key Actions
| Action | Purpose |
|--------|---------|
| `pick up OBJ` | Acquire tool if not in inventory |
| `use OBJ on OBJ` | Perform tool interaction |
| `look around` | Confirm target identity and location |

## Example
**Task:** Measure the temperature of a metal fork using a thermometer.

1. `pick up thermometer` — ensure tool is in inventory
2. `use thermometer on metal fork`
3. Observation: "a thermometer, currently reading a temperature of 72 degrees celsius"

## Important Notes
* The tool must be in your inventory before use. The `pick up` action is a prerequisite.
* Use exact object names as they appear in the environment description.
* Some tools produce measurable outputs (thermometer readings); others produce state changes (shovel digs soil). Interpret accordingly.
