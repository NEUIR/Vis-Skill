---
name: scienceworld-container-item-retriever
description: This skill retrieves a specified item from a container by using the 'pick up' action. It should be triggered when the agent has identified a needed item inside an open container (e.g., a jar, box, or hive) and must transfer it to inventory for later use. The skill handles the object transfer, assuming the container is already accessible, and outputs confirmation that the item is now in the agent's inventory.
---
# Skill: Retrieve Item from Container

## Purpose
Transfer a specified item from an open, accessible container into the agent's inventory using the `pick up` action.

## Trigger Conditions
Use this skill when **all** of the following are true:
1.  The agent has identified a specific item needed for a task.
2.  The item is located inside a container (e.g., jar, box, hive, closet).
3.  The container is already open and accessible (no `open` action required).
4.  The agent's immediate goal is to possess the item for later use.

## Core Action
Execute the `pick up <ITEM>` action, where `<ITEM>` is the exact name of the target object as observed in the environment.

## Execution Workflow
1.  **Verify Context:** Ensure the target item is present in the current room's observation and is inside an open container.
2.  **Execute Transfer:** Perform the `pick up <ITEM>` action.
3.  **Confirm Success:** The expected observation is: `"You move the <ITEM> to the inventory."` If a different result occurs, consult the troubleshooting guide.

## Important Notes
*   **Pre-condition:** This skill assumes the container is already open. If the container is closed, you must use the `open` action first.
*   **Ambiguity:** If multiple identical items exist (e.g., "avocado seed, avocado seed, avocado seed"), the `pick up` action will typically retrieve one. The skill does not handle selecting a specific instance.
*   **Post-condition:** After successful execution, the item is in the agent's inventory and can be used in subsequent steps.

## Example
**Task:** Retrieve an avocado seed from a jar for planting.
1. `look around` — observe: "a jar (containing an avocado seed)"
2. `pick up avocado seed`
3. Expected observation: "You move the avocado seed to the inventory."
