---
name: alfworld-tool-user
description: Use when the agent needs to apply a tool to a target object in ALFWorld to accomplish an interaction such as cleaning, heating, cooling, or examining. This skill handles locating both the tool and target object, then executing the correct environment action (e.g., `clean`, `heat`, `cool`, `use`) to progress the task.
---
# Skill: Tool User for ALFWorld

## When to Use
Trigger this skill when:
1. A task requires applying a tool to an object (e.g., cleaning a knife with a sinkbasin, heating food with a microwave)
2. Both the tool and target object are in the agent's possession or within reach
3. The agent knows which interaction verb to use for the task context

## Core Workflow

### 1. Identify Requirements
- Parse the task to determine:
  - **Target object**: The item to be acted upon (e.g., `knife 1`, `potato 2`)
  - **Required tool**: The appliance or instrument needed (e.g., `sinkbasin 1`, `microwave 1`)
  - **Interaction type**: The action verb (`clean`, `heat`, `cool`, `use`, `examine`)

### 2. Locate and Acquire
- **Target object**: If not in inventory, navigate to its location and `take {object} from {receptacle}`
- **Tool**: Navigate to the tool's location (tools are typically stationary appliances)

### 3. Execute Interaction
Apply the tool to the object using the correct action syntax:

| Interaction | Action Command | Example |
|-------------|---------------|---------|
| Clean | `clean {obj} with {tool}` | `clean knife 1 with sinkbasin 1` |
| Heat | `heat {obj} with {tool}` | `heat potato 2 with microwave 1` |
| Cool | `cool {obj} with {tool}` | `cool apple 1 with fridge 1` |
| Examine | `examine {obj}` | `examine book 3` |
| Use | `use {tool}` | `use desklamp 1` |

### 4. Verify Outcome
- Check the environment observation after the action
- A successful interaction changes the object state (e.g., `knife 1` becomes clean)
- If the observation says "Nothing happened," the action was invalid

## Error Handling
- **"Nothing happened"**: Reassess whether the agent is at the correct location and holding the correct object. Verify the action verb matches the task context.
- **Wrong tool**: If the tool does not match the interaction type, search for the correct appliance (e.g., use `sinkbasin` for cleaning, not `bathtub`).
- **Object not held**: The agent must be holding the target object before most interactions. Use `take` first.

## Example

**Task:** "Clean the knife and put it in the drawer."

```
> go to countertop 1
On the countertop 1, you see a knife 1, a saltshaker 2.
> take knife 1 from countertop 1
You pick up the knife 1 from the countertop 1.
> go to sinkbasin 1
On the sinkbasin 1, you see nothing.
> clean knife 1 with sinkbasin 1
You clean the knife 1 using the sinkbasin 1.
```

**Result:** `knife 1` is now clean and ready for the next step (storing).
