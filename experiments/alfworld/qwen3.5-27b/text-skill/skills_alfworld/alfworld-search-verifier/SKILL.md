---
name: alfworld-search-verifier
description: Re-examines previously visited locations to confirm the absence of a target object or to check for overlooked items. Use when an initial search fails to find enough objects or when double-checking is required before concluding task failure. Systematically revisits receptacles, re-opens closed containers, and re-inspects contents to ensure no viable location was missed.
---
# Instructions

**Trigger Condition:** Use this skill when an initial search for a target object (e.g., `cellphone`) has been performed but the required quantity has not been found, or when you need to verify that no viable location was missed before task failure.

## Core Procedure

1.  **Initialize Tracking:** Maintain a mental or explicit list of all receptacles (`recep`) you have already visited during your initial search (e.g., `sidetable 1`, `shelf 1-8`, `drawer 1`, `drawer 2`, `desk 1`).

2.  **Systematic Revisit:** Do not search new, unvisited locations. Instead, systematically return to each previously visited receptacle in a logical order (e.g., by proximity or by the original search sequence).
    *   **Action:** Use `go to {recep}` to navigate.
    *   **For Closed Receptacles:** If a receptacle was closed during the initial search and you opened it, it may be closed again. Re-open it to verify its contents.
        *   **Action:** Use `open {recep}`.
    *   **For Open/Empty Receptacles:** Observe the contents again. The state may have changed, or an item may have been overlooked.
    *   **Action:** No action needed; read the `Observation`.

3.  **Verification Logic:** For each revisited location:
    *   If the target object is now present, retrieve it (`take {obj} from {recep}`) and proceed with the main task.
    *   If the location is confirmed to lack the target object, note it as thoroughly checked and move to the next location on your list.
    *   If you encounter a closed receptacle you previously assumed was empty, opening it is a critical verification step.

4.  **Conclusion:** After revisiting all locations on your list:
    *   **If the target object was found:** Integrate it into your main task plan.
    *   **If the target object was not found:** You can conclusively report that the object is not available in the searched area and adjust your task strategy accordingly (e.g., consider task failure or exploring a new, unsearched area).

## Example

**Scenario:** You need two cellphones but only found one. Revisiting previously searched locations.

```
Thought: I only found 1 cellphone but need 2. Let me revisit sidetable 1 which I checked earlier.
Action: go to sidetable 1
Observation: On the sidetable 1, you see a cellphone 2, a keychain 1.
Thought: Found cellphone 2! I must have overlooked it during the initial search.
Action: take cellphone 2 from sidetable 1
Observation: You pick up the cellphone 2 from the sidetable 1.
```

**Result:** The second cellphone was found by re-examining a previously visited location.

## Thought Process Template
When executing this skill, structure your reasoning as follows:
