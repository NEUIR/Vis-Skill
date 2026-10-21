---
name: scienceworld-task-interpreter
description: This skill parses a user's high-level scientific task in the ScienceWorld environment and extracts the core objective and target location. It should be triggered when a new task instruction is received, especially those involving finding, comparing, or manipulating objects. The skill interprets the query to identify the goal (e.g., 'find the animal with the shortest life span') and any specified locations (e.g., 'animals are in the outside location'), outputting a clear, actionable sub-goal for navigation or observation.
---
# Instructions

Activate this skill when the user provides a new task instruction for the ScienceWorld environment. Your primary function is to interpret the instruction and generate a clear, executable plan.

## 1. Parse the Task
Read the user's instruction carefully. Extract the following core components:
*   **Primary Objective:** What is the ultimate goal? (e.g., "find", "compare", "manipulate").
*   **Target Object/Subject:** What is the main focus of the task? (e.g., "animal with the shortest life span").
*   **Specified Location:** Is a location explicitly mentioned? (e.g., "animals are in the 'outside' location").

## 2. Formulate the Plan
Based on the parsed components, construct a plan. The standard plan structure is:
1.  **Navigate:** If a target location is specified and you are not there, teleport to it immediately.
2.  **Observe:** Upon arriving at the correct location, use `look around` to survey the environment and identify relevant objects.
3.  **Analyze & Execute:** Apply domain knowledge or comparative reasoning to the observed objects to fulfill the primary objective. Then, take the final required action (e.g., `focus on [OBJECT]`, `pick up [OBJECT]`).

## 3. Output the Interpretation
Before taking the first action, articulate your interpretation and plan in a **Thought** step. This confirms the skill has correctly parsed the task.
**Output Format:**
