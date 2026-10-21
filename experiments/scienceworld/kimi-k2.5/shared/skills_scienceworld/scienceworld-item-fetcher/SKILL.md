---
name: scienceworld-item-fetcher
description: Picks up a specified object from the environment and moves it into the agent's inventory. Use this skill when you identify a required tool or material (e.g., thermometer, metal pot) that needs to be collected for later use in an experiment or task. Takes an object name as input and confirms the item's transfer to inventory.
---
# Skill: Fetch and Inventory Item

## Purpose
Use this skill when you need to acquire a specific object for a task. It handles locating the object in the current environment and moving it into your inventory for later use.

## Core Workflow
1.  **Identify Need:** Determine which object is required for your current or upcoming task (e.g., "thermometer", "metal pot").
2.  **Locate Object:** Ensure you are in the correct room. Use `look around` to survey the environment and confirm the object is present and accessible.
3.  **Execute Fetch:** Use the `pick up` action on the target object.
4.  **Verify Success:** Confirm the system observation states the object has been moved to your inventory.

## Key Principles
*   **Pre-check:** Always `look around` first to verify the object's presence and location (e.g., on a counter, in a container).
*   **Direct Action:** Use the simple `pick up OBJ` command. Do not overcomplicate the sequence.
*   **Inventory Focus:** After picking up the object, you may `focus on` it in your inventory to signal readiness for the next step in your main task.
*   **Error Handling:** If `pick up` fails, the object might be inside a closed container. Ensure all relevant containers are open first.

## Example from Trajectory
**Scenario:** Needing a thermometer to measure temperature.
1.  Observation: `a thermometer, currently reading a temperature of 10 degrees celsius` is seen in the kitchen.
2.  Action: `pick up thermometer`
3.  Result: `You move the thermometer to the inventory.`
4.  (Optional) Action: `focus on thermometer in inventory`

## Input
*   **Object Name:** The name of the object to fetch, as it appears in the environment observations (e.g., "thermometer", "metal pot").

## Output
*   A confirmation observation that the object is now in the agent's inventory.
