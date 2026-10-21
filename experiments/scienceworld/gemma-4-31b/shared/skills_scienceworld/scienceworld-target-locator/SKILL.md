---
name: scienceworld-target-locator
description: This skill determines the most likely location for a target object based on domain knowledge and environmental clues. Use when the agent needs to find a specific item (like an animal) but it is not in the current room. It analyzes the environment description and suggests a room to teleport to for further investigation.
---
# Skill: Target Locator

**Trigger:** When the agent needs to find a specific object (e.g., an animal, a tool, a chemical) and a preliminary `look around` in the current room does not reveal it.

## Procedure

1. Classify the target object by type.
2. Map to the most probable room using these heuristics:
   | Object Type | Likely Room(s) |
   |-------------|---------------|
   | animal | outside, garden |
   | tool/wire/battery | workshop |
   | food/cooking item | kitchen |
   | chemical/substance | lab, foundry |
   | plant/seed | garden, greenhouse |
   | container/box | workshop, kitchen |
3. Execute: `teleport to <ROOM>`
4. `look around` to verify the target is present. If not, try the next likely room.

## Example Flow (From Trajectory)
*   **Task:** "find a(n) animal."
*   **Trigger:** `look around` in hallway shows no animal.
*   **Analysis:** Target type = `animal`.
*   **Heuristic Check:** Animals are most commonly found `outside`.
*   **Context Check:** A door to `outside` is not listed in the hallway observation.
*   **Decision:** The highest-probability room `outside` is accessible via teleport.
*   **Output:** `teleport to outside`
