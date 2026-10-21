---
name: alfworld-receptacle-opener
description: This skill opens a closed receptacle to allow access to its contents or to enable interaction with it. It should be triggered when the agent encounters a closed receptacle (e.g., 'The fridge 1 is closed.') that must be opened to proceed with a task, such as cooling an item with a fridge or taking an object from inside a cabinet. The skill outputs the 'open' action on the specified receptacle.
---
# Skill: Open Closed Receptacle

## Trigger Condition
Use this skill when the environment observation explicitly states that a receptacle is **closed** and opening it is necessary to proceed with your current task. The typical observation pattern is: `"The <receptacle_name> is closed."`

## Action
Execute the `open` action on the identified receptacle.

## Procedure
1.  **Identify the Receptacle:** From the observation, extract the name of the closed receptacle (e.g., `fridge 1`, `cabinet 12`, `safe 1`).
2.  **Execute Action:** Output the action in the format: `Action: open <receptacle_name>`.

## Example from Trajectory
*   **Observation:** `"The fridge 1 is closed."`
*   **Thought:** `The fridge is closed. I need to open it to access the cooling compartment.`
*   **Action:** `Action: open fridge 1`

## Notes
*   This is a fundamental, low-level skill for enabling access. After opening the receptacle, you may need to use other skills (e.g., `take`, `cool`, `put`) to complete your objective.
*   Do not use this skill if the receptacle is already described as open or if its state is not mentioned.
