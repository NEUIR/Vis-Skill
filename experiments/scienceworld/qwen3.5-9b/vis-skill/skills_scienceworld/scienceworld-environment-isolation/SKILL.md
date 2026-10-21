---
name: scienceworld-environment-isolation
description: Use when you need to isolate a space (like a greenhouse) by closing doors or openings to create a contained environment. Trigger before starting pollination, temperature-sensitive experiments, or other environmental-sensitive tasks that require controlled conditions. Modifies room connectivity to optimize conditions for specific processes.
---
# Instructions

## When to Use
Trigger this skill when you need to:
1.  **Optimize Pollination:** Isolate a room (like a greenhouse) to concentrate pollinators (e.g., bees) and prevent them from leaving, thereby increasing the chance of successful cross-pollination between plants.
2.  **Control Environmental Factors:** Create a stable, contained space to manage variables like temperature, humidity, or airflow for a sensitive process.
3.  **Prevent Contamination or Interference:** Seal off a room to protect an ongoing experiment or task from external disturbances.

## Core Action Sequence
1.  **Assess the Environment:** Use `look around` to identify all doors, windows, or other openings in the current room.
2.  **Execute Isolation:** For each identified opening (e.g., "door to the outside", "door to the hallway"), issue a `close` command.
    *   **Example:** `close door to outside`
    *   **Example:** `close door to hallway`
3.  **Verify Closure:** Perform a final `look around` to confirm all intended openings are now closed and the room is successfully isolated.

## Key Principles
*   **Completeness:** Ensure *all* exits are closed for full isolation. An open door can defeat the purpose.
*   **Context-Specific:** The specific doors to close depend entirely on the room's layout. Always verify via `look around`.
*   **Post-Isolation:** After isolation, monitor the contained process (e.g., plant growth) and use `wait` actions as needed to allow time for the optimized conditions to take effect.

## Notes
*   This skill modifies the **connectivity** of a space, not its internal state (like temperature controls).
*   The primary observed benefit in the trajectory was enhancing bee-mediated pollination by preventing bee escape.
*   Reversal (re-opening doors) is not part of this skill's core function but can be done using standard `open` commands if needed later.

## Example

**Scenario:** You need to cross-pollinate two plants in the greenhouse using bees.

1. You have placed both plants and a bee jar in the `greenhouse`.
2. Run `look around` — you see: "door to outside (open)", "door to hallway (open)".
3. Run `close door to outside` — output: "You close the door to outside."
4. Run `close door to hallway` — output: "You close the door to hallway."
5. Run `look around` — confirm both doors now show as closed. The greenhouse is fully isolated.
6. Release the bee by running `open bee jar`. The bee cannot escape and will pollinate both plants.
7. Run `wait` several times to allow pollination to complete.
