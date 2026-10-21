---
name: scienceworld-material-classifier
description: This skill makes a determination about a material's property (e.g., conductivity) based on environmental cues or domain knowledge when direct testing fails. Trigger it when experimental actions are invalid or unavailable, requiring a logical inference. It uses observed object properties and common-sense reasoning to classify the material and decide its final disposition.
---
# Material Classification Skill

## When to Use
Activate when direct experimental testing of a material property (conductivity, magnetism, etc.) fails or equipment is unavailable, and you need to classify the material by inference to complete a sorting task.

## Procedure
1. `focus on <OBJECT>` — identify the target and note its material composition.
2. Attempt direct testing if equipment exists (e.g., `connect <OBJECT> terminal 1 to <WIRE> terminal 2`).
3. If testing fails, infer the property from the object's material. Consult `references/material_properties.md` for lookup.
4. `move <OBJECT> to <CONTAINER>` — place in the appropriate classification container.
5. `look at <CONTAINER>` — verify the object was placed correctly.

## Example
**Task:** Classify a glass jar for electrical conductivity when the circuit test is unavailable.
1. `focus on glass jar`
2. `connect glass jar terminal 1 to yellow wire terminal 2` — action fails (invalid connection).
3. Inference: glass is an electrical insulator → non-conductive.
4. `move glass jar to orange box`
5. `look at orange box` — observation: "containing a glass jar" — classification complete.
