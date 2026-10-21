---
name: scienceworld-room-navigator
description: Teleports the agent to a specified room within the ScienceWorld environment. Use when the agent needs to move between different locations to locate objects or access specific facilities. The input is a target room name, and the output is the agent arriving at that room, ready for further interaction.
---
# Instructions

Use this skill to instantly move the agent to a different room. This is the primary method of navigation.

## When to Use
*   The agent needs to find an object known to be in a specific room (e.g., "metal pot is located around the kitchen").
*   The agent needs to access a facility available only in a specific room (e.g., a workshop for building circuits, a foundry for smelting).
*   The agent has completed tasks in one room and needs to proceed to the next relevant location.

## How to Use
1.  **Identify the Target Room:** Determine the exact name of the destination room from the task context or environment description (e.g., `kitchen`, `workshop`, `foundry`).
2.  **Execute Teleport:** Use the `teleport to <ROOM_NAME>` action.
3.  **Verify Arrival:** The observation will confirm the teleport was successful (e.g., "You teleport to the kitchen."). You may then proceed with `look around` to survey the new location.

## Important Notes
*   Teleportation is a single-step action. There is no need for intermediate movement.
*   All standard room connections (doors) remain, but teleportation bypasses them.
*   After teleporting, the agent's inventory and any held items remain unchanged.

## Example
**Task:** Move to the kitchen to find a metal pot.
1. `teleport to kitchen`
2. Observation: "You teleport to the kitchen."
3. `look around` to survey the room and locate the metal pot.
