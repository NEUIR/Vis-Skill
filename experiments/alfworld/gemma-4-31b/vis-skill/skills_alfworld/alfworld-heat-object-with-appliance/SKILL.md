---
name: alfworld-heat-object-with-appliance
description: Uses a heating appliance (microwave, stoveburner, oven) to apply heat to a specified object. Use when the task requires warming or cooking an item (e.g., "heat some egg", "warm the mug") and a heating appliance is available. Takes the object name and appliance name as input and outputs the object in a heated state, ready for placement at the task's target location.
---
# Skill: Heat Object with Appliance

## Purpose
Heat a specified object using a designated kitchen appliance to change its state (e.g., from cold to hot, raw to cooked) as required by a task.

## Core Workflow
Follow this sequence of actions. Use the bundled `validate_and_plan.py` script to check for common preconditions before starting.

1.  **Navigate & Locate:** Go to the location of the target object.
2.  **Acquire Object:** Take the object from its current receptacle.
3.  **Navigate to Appliance:** Go to the specified heating appliance.
4.  **Prepare Appliance:**
    *   If the appliance is closed, open it.
    *   If the appliance contains other items, you may need to remove them (context-dependent). The default action is to close it and proceed, as some environments abstract this step.
5.  **Execute Heating:** Use the `heat {object} with {appliance}` action.
6.  **Complete Task:** Navigate to the final target location (e.g., a dining table) and put the heated object there.

## Key Decision Points
*   **Appliance State:** If the appliance is occupied, assess if the task implies clearing it. The trajectory shows proceeding without removal is sometimes valid.
*   **Object Location:** The object may not be at the initial location. Be prepared to search other receptacles (e.g., fridge, countertop, cabinet) if not found.
*   **Alternative Appliances:** If the primary appliance (e.g., microwave) is unavailable or broken, consider alternatives like a stoveburner.

## Example

**Task:** "Heat some egg and put it on the diningtable."

**Input:** `object: egg 1`, `appliance: microwave 1`, `target: diningtable 1`

**Sequence:**
1. `go to fridge 1` → Observation: "You are at fridge 1."
2. `open fridge 1` → Observation: "You open the fridge 1. You see egg 1, ..."
3. `take egg 1 from fridge 1` → Observation: "You pick up the egg 1 from the fridge 1."
4. `go to microwave 1` → Observation: "You are at microwave 1."
5. `heat egg 1 with microwave 1` → Observation: "You heat the egg 1 using the microwave 1."
6. `go to diningtable 1` → Observation: "You are at diningtable 1."
7. `put egg 1 in/on diningtable 1` → Observation: "You put the egg 1 in/on the diningtable 1."

**Output:** The egg 1 is heated and placed on the diningtable 1. Task complete.

## Bundled Resources
*   `scripts/validate_and_plan.py`: A utility to check the initial environment state against the skill's prerequisites.
*   `references/common_heating_appliances.md`: A list of typical appliances and their properties in the ALFWorld environment.
