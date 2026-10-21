---
name: scienceworld-inventory-manager
description: Handles picking up objects from the environment into the agent's inventory or moving them out to containers. Use this skill when you need to acquire an object for later use (pick up) or place an object into a specific container as part of a task requirement (move). Takes an object name and location as input and confirms the transfer to or from inventory.
---
# Skill: Inventory Manager

## Purpose
Manage the agent's inventory: pick up objects from the environment or place them into containers.

## Core Workflow
1. **Locate Object:** Use `look around` to confirm the object is present and note its exact name.
2. **Acquire:** `pick up OBJ` to move the object into your inventory.
3. **Place (if needed):** `move OBJ to OBJ` to transfer the object to a container.
4. **Verify:** Check the observation feedback to confirm the action succeeded. If it fails, verify exact object names with `look around`.

## Key Actions
| Action | Purpose |
|--------|---------|
| `look around` | Find objects and confirm names |
| `pick up OBJ` | Move object to inventory |
| `move OBJ to OBJ` | Place object into a container |

## Examples

**Acquiring an object:**
1. `look around` — see "a metal pot containing nothing" in the kitchen
2. `pick up metal pot containing nothing in kitchen`
3. Result: "You move the metal pot to the inventory."

**Placing an object:**
1. `move metal pot to blue box`
2. Result: Object transferred to the blue box.

## Important Notes
* Some objects include state descriptions in their name (e.g., "metal pot containing nothing"). Use the full name as shown in the environment.
* Use container color and name as specified in the task (e.g., `blue box`, `orange box`).
* Inventory capacity is limited — plan pick up and place actions to avoid conflicts.
