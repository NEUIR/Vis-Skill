---
name: alfworld-receptacle-finder
description: Searches for a suitable empty or appropriately occupied receptacle (like a shelf or table) to place an object. Use when you are holding an object that needs to be stored or placed and must find a receptacle that meets the placement criteria. Examines candidate receptacles by navigating to and inspecting each one until a suitable location is found.
---
# Instructions

**Trigger:** Use this skill when you have an object in your inventory that needs to be stored or placed, and you must find a suitable receptacle (e.g., shelf, table, basin).

## Core Procedure
1.  **Identify Target Object:** Confirm the object you are holding and its state (e.g., `soapbar 1`, `clean`).
2.  **Survey Environment:** Use `go to {recep}` actions to navigate to and visually inspect candidate receptacles from the observation list.
3.  **Evaluate Candidates:** For each receptacle, assess its suitability:
    *   **Primary Goal:** Find an **empty receptacle**.
    *   **Fallback:** If no empty receptacle is available, identify one that can logically accommodate the object without conflict (e.g., a shelf with a tissuebox may still have space for a soapbar).
4.  **Select & Place:** Navigate to the chosen suitable receptacle and execute `put {obj} in/on {recep}`.

## Key Considerations
*   **Object State:** Ensure the object meets any required pre-conditions (e.g., `clean`) before searching for a receptacle. Use skills like `clean` if necessary.
*   **Efficiency:** Prioritize receptacles that are logically appropriate for the object (e.g., a `shelf` for a `soapbar` over a `toilet`).
*   **Observation Parsing:** Carefully read the observation after each `go to` action. The line "On the {recep}, you see..." lists its contents. "you see nothing." indicates it is empty.
*   **Invalid Actions:** If an action fails ("Nothing happened"), re-evaluate your target and try a different receptacle or confirm the object is in your inventory.

## Example

**Scenario:** You are holding `soapbar 1` (clean) and need to find an empty shelf to place it.

```
Thought: I need to find an empty shelf for the soapbar. I'll check shelf 1 first.
Action: go to shelf 1
Observation: On the shelf 1, you see a tissuebox 1.
Thought: Shelf 1 has items. Let me try shelf 2.
Action: go to shelf 2
Observation: On the shelf 2, you see nothing.
Action: put soapbar 1 in/on shelf 2
Observation: You put the soapbar 1 in/on the shelf 2.
```

**Result:** Found an empty shelf and placed the soapbar on it.

## Bundled Reference
For a detailed example including handling object pre-cleaning and sequential shelf evaluation, refer to `references/example_trajectory.md`.
