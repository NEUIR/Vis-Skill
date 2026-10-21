---
name: scienceworld-recipe-retriever
description: This skill locates and acquires a recipe or instruction document by using 'pick up' on the recipe object. Use when the task involves following a specific procedure (e.g., crafting, mixing) and the agent needs to obtain the written instructions. The skill assumes the recipe is visible in the current room and moves it to the inventory, allowing the agent to read it later to understand required ingredients and steps.
---
# Skill: Recipe Retriever

## Purpose
Acquire a visible recipe or instruction document from the current environment and store it in your inventory for later reference.

## Trigger Conditions
Use this skill when:
1. The task involves following a specific procedure (e.g., crafting, mixing, building).
2. You are instructed to find or use a recipe.
3. You need written instructions to understand required ingredients and steps.

## Core Procedure
1. **Locate Recipe**: Use `look around` to survey the current room. Identify any object described as a "recipe," "instructions," "manual," or similar document.
2. **Acquire Recipe**: Use `pick up <recipe_object>` to move the recipe to your inventory.
3. **Verify Acquisition**: Confirm the recipe is now in your inventory before proceeding to the next task phase.

## Key Notes
- This skill assumes the recipe is visible and accessible in the current room.
- The skill ends once the recipe is successfully moved to inventory.
- Reading the recipe (using `read`) is a separate action that should be performed after acquisition.
- If the recipe is not found in the current room, you may need to explore other rooms first.

## Error Handling
- If `pick up` fails (e.g., "You cannot pick that up"), examine the object first to verify it's a recipe.
- If no recipe is found after looking around, expand your search to adjacent rooms.

## Example
**Task:** Obtain the recipe for crafting red paint.
1. `look around` — observe: "a recipe for red paint on the counter"
2. `pick up recipe for red paint`
3. Expected observation: "You move the recipe for red paint to the inventory."
4. Later: `read recipe for red paint` to learn the required ingredients.
