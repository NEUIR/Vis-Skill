---
name: alfworld-inventory-management
description: Use when the agent must collect and track multiple instances of the same object type in ALFWorld (e.g., "put two cellphone in bed"). This skill maintains a count of collected versus needed objects, guides systematic searching through receptacles, and ensures each found object is placed at the target before searching for the next.
---
# Inventory Management Skill

## When to Use
Activate this skill when:
- Task requires collecting **multiple instances** of the same object type (e.g., "put two cellphone in bed")
- You need to track progress toward a quantity-based goal
- Searching through multiple locations systematically

## Core Workflow

### 1. Initialize Inventory
- Parse the task description to identify:
  - **Target object type** (e.g., "cellphone")
  - **Required quantity** (e.g., "two" = 2)
  - **Target receptacle** (e.g., "bed 1")
- Initialize counters: `collected = 0`, `needed = <quantity>`
- Create empty list for searched locations

### 2. Systematic Search Pattern
Follow this search priority:
1. **Visible surfaces** (desks, dressers, beds, countertops) - check first
2. **Closed containers** (drawers, cabinets, safes) - open and inspect
3. **Less common locations** (shelves, side tables, garbage cans)
4. **Return to known locations** if inventory incomplete

**Critical Rule:** After finding an object, immediately place it at the target location before searching for the next one. Do not attempt to carry multiple objects simultaneously.

### 3. Action Decision Logic
Use this decision tree at each step:

```
Is target object visible in current observation?
├── YES → Take it, go to target receptacle, put it down
│         └── Increment collected counter
│             ├── collected == needed → TASK COMPLETE
│             └── collected < needed → Continue searching
└── NO → Have all receptacles been searched?
          ├── YES → Revisit receptacles (objects may have been missed)
          └── NO → Go to next unsearched receptacle
```

### 4. Per-Object Cycle
For each object instance found, follow this exact sequence:
1. `take {object} from {current_receptacle}`
2. `go to {target_receptacle}`
3. `put {object} in/on {target_receptacle}`
4. Update counter: `collected += 1`
5. If `collected < needed`, navigate to next unsearched receptacle

## Example

**Task:** "Put two cellphone in bed 1."

```
> go to desk 1
On the desk 1, you see a cellphone 2, a pen 1.
> take cellphone 2 from desk 1
You pick up the cellphone 2 from the desk 1.
> go to bed 1
On the bed 1, you see a pillow 1.
> put cellphone 2 in/on bed 1
You put the cellphone 2 in/on the bed 1.
[collected: 1/2]
> go to dresser 1
On the dresser 1, you see a cellphone 3, a keychain 1.
> take cellphone 3 from dresser 1
You pick up the cellphone 3 from the dresser 1.
> go to bed 1
On the bed 1, you see a cellphone 2, a pillow 1.
> put cellphone 3 in/on bed 1
You put the cellphone 3 in/on the bed 1.
[collected: 2/2 — TASK COMPLETE]
```

## Error Handling
- **Object not at expected location**: Mark location as searched, proceed to next receptacle
- **"Nothing happened"**: The action syntax may be wrong; verify object name and receptacle
- **Counter mismatch**: Re-examine the target receptacle to confirm how many objects are already placed
