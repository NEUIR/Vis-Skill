---
name: scienceworld-living-entity-identifier
description: Analyzes room observations to identify potential living things (e.g., eggs, plants, animals) among listed objects. Use this skill when a task involves finding, focusing on, or interacting with a living thing, biological entity, or organism. Processes observation text, flags candidate living items based on domain knowledge, and outputs a focused target for subsequent actions like focus on or pick up.
---
# Skill: Living Entity Identifier

## Purpose
Identify living things (eggs, plants, animals) from room observations and focus on them for task progression.

## Core Workflow
1. **Survey Room:** `look around` to get the observation text listing all objects.
2. **Identify Candidates:** Scan the object list for living entity indicators:
   - **Animals:** dove, giant tortoise, bee, frog, fish
   - **Eggs:** dove egg, chicken egg, turtle egg
   - **Plants:** flower, tree, moss, fern, algae
   - **Other biological:** mushroom, seed, pollen
3. **Focus on Target:** `focus on [IDENTIFIED_OBJECT]` to signal task progress.
4. **Transport (if needed):** `pick up [OBJECT]` then `move [OBJECT] to [CONTAINER]`.

## Key Actions
| Action | Purpose |
|--------|---------|
| `look around` | Survey room for objects |
| `teleport to LOC` | Move to rooms with biological likelihood |
| `focus on OBJ` | Signal identification to task system |
| `pick up OBJ` | Acquire entity for transport |

## Example
**Task:** "Find a living thing in the environment."

1. `teleport to outside`
2. `look around` — observation lists: "a dove egg", "a rock", "soil"
3. Identify "dove egg" as the living entity candidate
4. `focus on dove egg`
5. If transport required: `pick up dove egg` then `move dove egg to blue box`

## Important Notes
* Prioritize explicit living things (e.g., "dove egg," "giant tortoise") over ambiguous substances (e.g., "air," "water").
* If the current room lacks candidates, `teleport to` rooms with higher biological likelihood: `outside`, `greenhouse`, `bedroom`.
* Always use `focus on` immediately after identification — it signals task progress.
