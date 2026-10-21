---
name: scienceworld-substance-preparator
description: Use when you need to transfer a target substance into an appropriate container for processing (e.g., a pot for heating, a beaker for mixing). Trigger after acquiring the substance and before setting up an apparatus. Selects a suitable empty container and moves the substance into it.
---
# Substance Preparation Skill

## Purpose
Prepare a target substance for subsequent processing (e.g., heating, mixing) by transferring it from its current location into an appropriate, empty container.

## When to Use
1. **Trigger:** After successfully locating and focusing on the target substance.
2. **Prerequisite:** The substance must be accessible (e.g., in inventory, in an open container).
3. **Next Step:** This skill should be completed **before** setting up any apparatus (e.g., activating a stove, connecting electrical components).

## Core Procedure
1.  **Identify Target Substance:** Confirm the exact name and location of the substance to be prepared.
2.  **Scan Environment for Containers:** Look for empty, process-appropriate containers in the current room. Priority order:
    *   **Primary:** Metal pots, beakers, or bowls (for heating/mixing).
    *   **Secondary:** Ceramic/glass cups, jars (if primary not available).
    *   **Avoid:** Containers already holding items or substances.
3.  **Select & Validate Container:** Choose the most suitable empty container. If uncertain, `examine` it to confirm it's empty.
4.  **Execute Transfer:** Use the `move [SUBSTANCE] to [CONTAINER]` action.
5.  **Verify:** Briefly check the container's contents to confirm the transfer was successful.

## Key Considerations
*   **Container State:** All containers are pre-opened. Do not use `open` or `close` actions.
*   **Efficiency:** Prefer containers in the same room to minimize `teleport` use.
*   **Substance Integrity:** If the substance is temperature-sensitive (e.g., chocolate in a fridge), moving it to a room-temperature container is part of preparation.
*   **Error Handling:** If the transfer fails (e.g., container not found), `look around` again and consult the bundled reference for common container names.

## Example

**Scenario:** You need to melt chocolate as part of a heating task.

1. You have already located `chocolate` in the `fridge` in the kitchen.
2. Run `look around` to scan for containers. You see: `metal pot` (empty, on counter), `glass beaker` (empty, on shelf), `ceramic cup` (contains water).
3. Select `metal pot` as the best option (metal, empty, suitable for heating).
4. Run `examine metal pot` to confirm it is empty.
5. Run `move chocolate to metal pot`.
6. Run `examine metal pot` — output confirms: "metal pot (containing chocolate)".

**Proceed to the next skill (e.g., apparatus setup) only after this verification.**
