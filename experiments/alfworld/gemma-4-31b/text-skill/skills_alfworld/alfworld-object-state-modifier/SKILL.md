---
name: alfworld-object-state-modifier
description: Uses an appliance to change the state of an object (cooling, heating, or cleaning). Use when the task requires altering an object's temperature or cleanliness using a specific device (e.g., cooling with a fridge, heating with a microwave, cleaning with a sinkbasin). Takes the object, target state, and appliance as inputs and executes the corresponding modifier action.
---
# Instructions

Change an object's state (cool, heat, or clean) using a household appliance. You must be holding the target object before executing the state change.

## Workflow
1. **Navigate to appliance:** `go to {appliance}` (e.g., `go to fridge 1`)
2. **Prepare if closed:** If observation says appliance is closed, execute `open {appliance}` -- verify it opens
3. **Execute state change:** `{action} {obj} with {appliance}` -- verify observation confirms the change
4. **Proceed:** Continue with the next task step (e.g., placing the modified object)

## Action Mapping
| State Change | Action | Typical Appliance |
|---|---|---|
| Cool | `cool {obj} with {appliance}` | `fridge 1` |
| Heat | `heat {obj} with {appliance}` | `microwave 1`, `stoveburner 1` |
| Clean | `clean {obj} with {appliance}` | `sinkbasin 1` |

## Error Recovery
- "Nothing happened": Check (1) you are holding the object, (2) appliance is open/ready, (3) object and appliance names are correct
- Appliance occupied: remove existing items first with `take {item} from {appliance}`

## Example

**Scenario:** You need to cool a tomato using the fridge.

```
Thought: I'm holding tomato 1 and need to cool it. I'll go to the fridge.
Action: go to fridge 1
Observation: The fridge 1 is closed.
Action: open fridge 1
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 1.
Action: cool tomato 1 with fridge 1
Observation: You cool the tomato 1 using the fridge 1.
```

**Result:** The tomato is now in a cooled state. Proceed with the next task step (e.g., placing it on a countertop).
