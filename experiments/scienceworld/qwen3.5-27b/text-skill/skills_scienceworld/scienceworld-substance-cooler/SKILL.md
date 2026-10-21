---
name: scienceworld-substance-cooler
description: This skill initiates the cooling of a substance by moving it into a cooling appliance like a freezer. Use when a task requires lowering the temperature of a specific material to observe phase changes. The skill takes the substance (often in a container) and the target appliance as inputs, using the 'move OBJ to OBJ' action. It outputs confirmation of the new location.
---
# Skill: Substance Cooler

## Purpose
Use this skill to begin the process of cooling a substance to potentially observe a phase change (e.g., freezing). The core action is to relocate the substance into a cooling appliance.

## When to Use
*   The task explicitly requires cooling a substance (e.g., "measure the melting point of mercury").
*   You have identified both the target substance (often inside a container like a cup or jar) and a suitable cooling appliance (e.g., freezer, ultra low temperature freezer).
*   The immediate goal is to lower the substance's temperature.

## Core Instruction
1.  **Locate Items:** Ensure you are in the room containing the target substance and the cooling appliance. Use `look around` if necessary.
2.  **Execute Move:** Perform the action `move <SUBSTANCE_CONTAINER> to <APPLIANCE>`.
    *   `<SUBSTANCE_CONTAINER>`: The object holding the substance (e.g., "glass cup (containing a substance called mercury)").
    *   `<APPLIANCE>`: The cooling device (e.g., "freezer").
3.  **Verify:** Use `look at <APPLIANCE>` to confirm the substance is now inside the appliance. Report this confirmation.

## Important Notes
*   **Prerequisite:** The appliance door must be open. The trajectory shows all containers are already open, so this should not require a separate `open` action.
*   **This is an Initialization Skill:** This skill only *starts* the cooling process. Subsequent skills or actions (like `use thermometer on <SUBSTANCE>`) are required to monitor temperature changes and observe the phase transition.
*   **Inputs:** The skill requires the names of the substance container and the target appliance as identified in the environment observations.
*   **Output:** A confirmation observation stating the substance is now inside the appliance.

## Example from Trajectory
**Trigger:** Task requires measuring the melting point of mercury.
**Action:** `move glass cup (containing a substance called mercury) to freezer`
**Verification:** `look at freezer` yields "In the freezer is: a glass cup (containing a substance called mercury)."
**Output:** "The mercury in its glass cup has been placed into the freezer to begin cooling."
