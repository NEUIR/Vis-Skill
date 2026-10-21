---
name: alfworld-object-locator
description: Use when the agent needs to find a specific object in ALFWorld that is not currently in inventory and whose location is unknown. This skill parses the environment observation, ranks receptacles by likelihood of containing the target object using common-sense reasoning, and outputs a navigation action to the most promising location.
---
# Skill: Object Locator for ALFWorld

## When to Use
Trigger this skill when:
1. Your goal requires a specific object (e.g., `knife`, `cellphone`, `apple`)
2. The object is not in your inventory
3. The current observation does not explicitly state the object's location

## Core Workflow

### 1. Parse the Environment
Extract all visible receptacles from the observation text. Typical ALFWorld receptacles include:
- **Surfaces**: countertop, desk, dresser, bed, shelf, sidetable, coffeetable, diningtable
- **Containers**: drawer, cabinet, safe, fridge, microwave, garbagecan
- **Appliances**: sinkbasin, bathtub, stoveburner, toaster

### 2. Rank by Likelihood
Use common-sense reasoning to prioritize where the target object is most likely found:

| Object Type | High-Probability Receptacles |
|-------------|------------------------------|
| Kitchen items (knife, spatula, pan) | countertop, drawer, diningtable, stoveburner |
| Food (apple, potato, tomato, bread) | fridge, countertop, diningtable, microwave |
| Bathroom items (sponge, cloth, soap) | sinkbasin, bathtub, cart, shelf |
| Electronics (cellphone, laptop, remote) | desk, sidetable, dresser, bed, coffeetable |
| Stationery (pen, pencil, book) | desk, shelf, drawer, sidetable |
| Lighting (candle, desklamp) | sidetable, shelf, desk, dresser |

### 3. Navigate and Search
For each candidate receptacle (in priority order):
1. `go to {receptacle}`
2. Read the observation — does it mention the target object?
   - **YES**: `take {object} from {receptacle}` — object found
   - **NO**: If the receptacle is closed, `open {receptacle}` and re-check
   - **Still NO**: Move to the next candidate receptacle

### 4. Track Searched Locations
Maintain a list of already-searched receptacles to avoid revisiting them. If all high-probability locations are exhausted, expand the search to remaining receptacles.

## Example

**Task:** "Clean the knife and put it in drawer."
**Observation:** "You are in the middle of a room. Looking quickly around you, you see a countertop 1, a drawer 1, a drawer 2, a fridge 1, a sinkbasin 1, a stoveburner 1."

```
> go to countertop 1
On the countertop 1, you see a knife 1, a saltshaker 2, a bread 1.
> take knife 1 from countertop 1
You pick up the knife 1 from the countertop 1.
```

**Result:** Target object `knife 1` located and acquired from `countertop 1`.

**Example — Object not at first location:**

```
> go to countertop 1
On the countertop 1, you see a saltshaker 2, a bread 1.
> go to drawer 1
The drawer 1 is closed.
> open drawer 1
You open the drawer 1. The drawer 1 is open. In it, you see a knife 1.
> take knife 1 from drawer 1
You pick up the knife 1 from the drawer 1.
```

## Error Handling
- **Object not found in any receptacle**: Re-check closed containers that may not have been opened. Some objects are only visible after opening.
- **Multiple instances**: If the task requires a specific instance (e.g., `knife 1` vs `knife 2`), verify the object identifier matches before taking it.
- **"Nothing happened"**: The `take` command may fail if the agent is not at the receptacle. Ensure navigation was successful before attempting to take.
