---
name: scienceworld-liquid-source-finder
description: This skill searches rooms for a liquid source (e.g., sink, toilet, jug) by sequentially surveying different locations. It should be triggered when the task requires obtaining a liquid (like water) and the agent does not have immediate access to it. The skill involves teleporting to candidate rooms, looking around, and identifying potential sources, guiding the agent toward the nearest available liquid.
---
# Skill: Liquid Source Finder

## When to Use
Activate this skill when your primary task requires a liquid (e.g., water, juice, paint) and you have confirmed it is not present in your current inventory or immediate vicinity.

## Core Procedure
1.  **Initiate Search:** If no liquid source is visible, begin a systematic room survey.
2.  **Prioritize Rooms:** Teleport to and survey rooms in this order of likelihood:
    *   **Kitchen** (high priority - contains sink, fridge, cups)
    *   **Bathroom** (high priority - contains sink, toilet, bathtub)
    *   **Greenhouse** (medium priority - contains sink, jugs)
    *   **Workshop** (medium priority - may contain specialized containers)
    *   **Art Studio** (low priority - may contain paint or water for cleaning)
    *   **Living Room, Bedroom, Hallway** (low priority - check for any containers)
3.  **Survey Each Room:**
    *   Use `look around` upon arrival.
    *   Visually scan the observation for liquid sources or containers that might hold liquid (e.g., `sink`, `toilet`, `jug`, `cup`, `bathtub`).
    *   If a source is found (e.g., a `sink`), proceed to obtain the liquid. If a container is found (e.g., a `jug`), `examine` it or `look at` it to check its contents.
4.  **Obtain the Liquid:**
    *   If the liquid is in a portable container (e.g., `cup containing water`), simply `pick up` the container.
    *   If the liquid is in a fixed source (e.g., `sink`, `toilet`):
        a. Ensure you have an empty container in your inventory. If not, find one (e.g., a `cup`).
        b. `move` your container to the source.
        c. `activate` the source if necessary (e.g., turn on the sink).
        d. `deactivate` the source once done.
        e. `pick up` the now-filled container.
5.  **Terminate Search:** Once the required liquid is successfully obtained and in your inventory, conclude the skill and return to the main task flow.

## Key Principles
*   **Efficiency:** Do not re-survey rooms unnecessarily. Keep mental note of rooms already visited.
*   **Container Management:** Always ensure you have a suitable container before attempting to collect liquid from a non-portable source.
*   **Observation Parsing:** Focus on object names and their described contents. Look for keywords like `water`, `juice`, or the name of your target liquid, and objects like `sink`, `toilet`, `jug`.

## Quick Reference
**Trigger:** Need liquid, don't have it.
**Action Loop:** Teleport -> `look around` -> Identify source -> Collect.
**Success:** Liquid is in inventory.
