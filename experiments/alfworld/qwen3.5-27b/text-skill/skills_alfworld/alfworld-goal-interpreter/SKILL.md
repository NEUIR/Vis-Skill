---
name: alfworld-goal-interpreter
description: Parses the natural language task goal to extract actionable sub-objectives and required objects. Trigger this skill whenever a new task is assigned to break down complex instructions into clear, sequential targets. It interprets phrases like 'look at X under Y' to identify target objects (pillow), reference objects (desklamp), and spatial relationships (under).
---
# Goal Interpretation Protocol

## 1. Parse the Task Statement
When a new task is assigned, immediately analyze the natural language instruction to extract its core components. Use the parsing script (`parse_goal.py`) to perform this analysis.

**Input:** The raw task string (e.g., "look at pillow under the desklamp").
**Output:** A structured dictionary containing:
- `primary_target`: The main object to interact with.
- `reference_object`: The object that defines a location or condition.
- `spatial_relation`: The preposition linking them (e.g., under, on, in).
- `action`: The verb defining the required interaction.

## 2. Generate Sub-Objectives
Based on the parsed components, formulate a clear, sequential plan. The plan must account for the spatial relationship.

**Example Logic:**
- **IF** relation is `under` → Sub-goal 1: Locate the `reference_object`. Sub-goal 2: Inspect the area beneath it for the `primary_target`.
- **IF** relation is `in` or `on` → Sub-goal 1: Locate the `reference_object`. Sub-goal 2: Check its contents/surface.

## 3. Identify Required Objects & Actions
Map the parsed objects to the available action types in the environment. The primary action verbs (look, take, use, etc.) from the task must be translated into the agent's available action set (go to, take, use, etc.).

**Critical Check:** If the `reference_object` is a container (drawer, fridge), ensure the plan includes the `open` action before inspection.

## 4. Execute & Adapt
Initiate the plan. After each action, monitor the observation. If the expected object is not found, or the action fails ("Nothing happened"), consult the fallback logic in the reference guide (`search_patterns.md`).

**Remember:** The agent must maintain object permanence. If an object is moved (e.g., a pillow is picked up), it is now in the agent's inventory and the spatial relationship is void.
