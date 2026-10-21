---
name: scienceworld-task-parser
description: Analyzes user instructions in ScienceWorld environments to extract specific task requirements and constraints. Use when receiving a new task to identify required objects, target locations, and action sequences before taking any environment actions.
---
# Instructions for ScienceWorld Task Parsing

When you receive a new task instruction in a ScienceWorld environment, follow this process to parse it into a structured plan.

## 1. Parse the Instruction
Immediately analyze the user's natural language instruction. Extract the following core components:
*   **Target Object Type:** Identify the category or description of the object to be manipulated (e.g., 'non-living thing', 'liquid', 'electrical component').
*   **Target Location:** Identify the final destination for the object, including the room and specific container (e.g., 'purple box in the workshop').
*   **Required Actions:** Infer the sequence of actions implied by verbs like "find", "focus on", "move", "pour", "mix", etc.

**Output your analysis as a concise thought.** Example: "Task requires finding a non-living object in the workshop and moving it to the purple box."

## 2. Survey the Environment
*   Use `look around` in your current room to get an inventory of visible objects, containers, and their states.
*   If the target location is a different room, use `teleport to LOC` to go there first, then `look around`.

## 3. Identify the Target Object
*   From the room description, identify objects matching the parsed **Target Object Type**.
*   If multiple candidates exist, select one that is clearly non-living, portable, and not part of a fixed apparatus (e.g., a wire, a light bulb, a battery). Avoid substances like 'air'.
*   Use `examine OBJ` or `look at OBJ` if you need more detail to confirm an object's properties.

## 4. Execute the Task Sequence
1.  **Signal Intent:** Use `focus on OBJ` on the identified target object. This explicitly marks the object for the task.
2.  **Perform Core Action:** Execute the primary action from the parsed instruction (e.g., `move OBJ to OBJ`, `pour OBJ into OBJ`).
3.  Use `wait` or `wait1` only if necessary to allow for state changes.

## Key Principles
*   **Efficiency:** All containers are pre-opened. Do not use `open` or `close` unless explicitly required.
*   **Directness:** Teleport directly to the target room. Do not explore unrelated rooms.
*   **Clarity:** Structure your internal reasoning using the "Thought:" prefix before each action, as shown in the trajectory.
*   **Verification:** If an initial `look around` is insufficient, a second `look around` is acceptable to confirm the environment state before proceeding.

## Example
**Task instruction:** "Find a non-living thing in the workshop and move it to the purple box."
1. **Parse:** Target = non-living object, Location = workshop, Container = purple box.
2. `teleport to workshop`
3. `look around` — observe: "a battery, a blue light bulb, an orange wire..."
4. Select: battery (non-living, portable).
5. `focus on battery`
6. `move battery to purple box`
