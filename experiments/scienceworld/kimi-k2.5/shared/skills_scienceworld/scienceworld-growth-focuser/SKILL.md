---
name: scienceworld-growth-focuser
description: Use when you have planted a seed or need to track a plant's growth stage (sprouting, flowering, reproduction). Applies the 'focus on' action to a specific plant or biological entity to signal intent and monitor its development. Trigger after planting or when you need to observe life cycle progression in the ScienceWorld environment.
---
# Skill: Growth Focuser

## Purpose
Apply the `focus on` action to a specific biological entity (e.g., a planted seed or growing plant) to signal your intent to monitor its development. This action is often a prerequisite for the environment to process growth stages or trigger life cycle progression events.

## When to Use
Use this skill when:
1. **After Planting:** You have just placed a seed or seedling into a suitable growth medium (e.g., soil with water).
2. **Stage Monitoring:** You need to check or trigger a progression in the plant's life cycle (e.g., from seed to sprout, sprout to mature plant, flowering, or reproduction).
3. **Intent Signaling:** The task requires you to explicitly indicate which entity you are observing or nurturing.

## Core Instruction
1.  **Identify the Target:** Locate the specific plant or seed you intend to focus on. Ensure it is in a valid state for growth (e.g., planted in soil with water).
2.  **Execute Focus:** Use the action: `focus on <PLANT_OBJECT> [in <CONTAINER>]`.
    *   Replace `<PLANT_OBJECT>` with the exact name of the biological entity (e.g., `avocado seed`).
    *   Use the optional `[in <CONTAINER>]` clause if the object is inside another object (e.g., `avocado seed in flower pot 1`). This is crucial for disambiguation.
3.  **Confirm & Wait:** Upon successful focus, you will receive a confirmation. Biological growth requires time. Use `wait` actions strategically after focusing to allow the simulated environment to process development stages.

## Key Considerations
*   **Disambiguation:** If multiple identical objects exist, the environment may prompt you to choose. Always specify the container (e.g., `in flower pot 1`) to avoid this.
*   **Prerequisites:** The `focus on` action typically follows preparatory steps like `teleport to <location>`, `pick up <item>`, and `move <seed> to <pot>`.
*   **Patience is Required:** Growth is not instantaneous. The `focus on` action signals intent, but you must follow it with `wait` actions to observe results.

## Example

**Scenario:** You need to grow an avocado plant from seed to maturity.

1. You have already moved `avocado seed` into `flower pot 1` which contains soil and water.
2. Run `focus on avocado seed in flower pot 1` — output: "You focus on the avocado seed."
3. Run `wait` — the environment processes one time step.
4. Run `focus on avocado seed in flower pot 1` again — output may now show: "avocado sprout" indicating growth progression.
5. Continue alternating `focus on` and `wait` until the plant reaches the desired life stage (sprout, mature plant, flowering, reproduction).

**Note:** Always include the `in <CONTAINER>` clause when multiple pots or seeds exist to avoid disambiguation prompts.

## Quick Reference
**Trigger Condition:** Seed/Plant is in a growth-ready state.
**Primary Action:** `focus on <PLANT_OBJECT> [in <CONTAINER>]`
**Next Steps:** Use `wait` to allow time for growth progression.
