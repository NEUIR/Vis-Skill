---
name: scienceworld-result-archiver
description: Places an object into a designated container (like a colored box) based on a test outcome. It should be triggered when the agent needs to finalize a task by storing an object according to a rule (e.g., conductive → blue box, non-conductive → orange box). The input is the object and the rule, and the output is the object moved to the correct container.
---
# Instructions
Use this skill to finalize a scientific test by archiving an object based on the observed result.

## When to Use
- You have just completed an experimental test (e.g., electrical conductivity, chemical reaction, physical property test).
- You have a clear rule mapping a specific test outcome to a specific destination container (e.g., "If property X is true, place in Container A; if false, place in Container B").
- The object to be archived and the destination containers are present in your current environment.

## Core Procedure
1.  **Verify Context:** Ensure you are in the correct room (typically the `workshop` or lab area) where the test was conducted and where the destination containers are located.
2.  **Confirm Test Result:** Observe the final state of your experimental apparatus to definitively determine the test outcome (e.g., "light bulb is on").
3.  **Apply Rule:** Map the confirmed outcome to the corresponding destination container as specified by the task rule.
4.  **Execute Archive:** Move the test object from its current location (inventory or room) into the correct container using the `move OBJ to CONTAINER` action.

## Key Principles
- **Direct Archiving:** Do not re-run the test. The skill is for archiving the result based on an *already observed* outcome.
- **Rule Adherence:** Strictly follow the provided mapping rule. Do not infer or create new rules.
- **Container Verification:** Before moving the object, visually confirm the target container exists in the room (use `look around` if uncertain).

## Example Rule Application
*Task Rule:* "If the metal pot is electrically conductive, place it in the blue box. If it is electrically nonconductive, place it in the orange box."
*Observation:* The blue light bulb in the circuit is **on**.
*Interpretation:* The metal pot is conductive.
*Action:* `move metal pot to blue box`

## Bundled Logic
For the specific, error-prone sequence of connecting a circuit to test electrical conductivity, use the bundled script `scripts/conductivity_test.py` as a reference. For all other test types, use the general instructions above.
