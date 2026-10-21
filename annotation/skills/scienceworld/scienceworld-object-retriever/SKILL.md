---
name: scienceworld-object-retriever
description: Acquires a specified object by moving it from the environment into the agent's inventory using 'pick up OBJ'. Use this skill when a task requires an object to be manipulated, tested, or transported and it is not yet in your inventory. Makes the object available for further actions like testing, measuring, or placing.
---
# Skill: Object Retriever

## Purpose
Acquire a target object from the environment and place it into your inventory for further use.

## When to Use
- You need an object for testing, measuring, or transporting.
- The object is visible in the room but not yet in your inventory.
- A downstream skill (e.g., tool-user, object-placer) requires the object in inventory.

## Core Workflow
1. **Navigate:** `teleport to LOC` if needed to reach the room containing the object.
2. **Survey:** `look around` to confirm the object is present and identify its exact name.
3. **Acquire:** `pick up OBJ` using the exact name from the environment description.
4. **Verify:** Check the observation feedback confirms "You move the [object] to the inventory."

## Example
**Task:** Retrieve a metal fork from the kitchen for conductivity testing.

1. `teleport to kitchen`
2. `look around` — see "a metal fork" on the counter
3. `pick up metal fork`
4. Observation: "You move the metal fork to the inventory."

## Important Notes
* Ensure you are in the correct room before attempting `pick up`. Use `teleport to LOC` if necessary.
* If `look around` does not reveal the object, use `examine` on containers or furniture (e.g., `examine table`).
* If a container is closed, use `open OBJ` first.
* After acquisition, the object is ready for the next skill in the task sequence.
