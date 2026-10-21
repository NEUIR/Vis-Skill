---
name: alfworld-task-verifier
description: Use when the agent needs to check whether an ALFWorld task objective has been met after completing a sub-action (e.g., placing an object). This skill parses the task goal, evaluates the latest environment observation, and outputs a verification decision — task complete, task incomplete, or action ineffective — to guide the next step.
---
# Skill: Task Verifier for ALFWorld

## When to Use
Trigger this skill when:
1. The agent has just completed a key sub-action (e.g., `put {obj} in/on {recep}`)
2. The agent needs to determine whether the overall task goal is satisfied
3. The agent must decide whether to continue searching or conclude the task

## Core Workflow

### 1. Parse the Task Goal
Extract from the original task description:
- **Target object(s)**: What needs to be found/placed (including quantity)
- **Target receptacle**: Where objects must end up
- **Required transformations**: Any cleaning, heating, or cooling steps

### 2. Analyze the Observation
Evaluate the most recent `Observation:` text from the environment:
- Look for evidence that required objects are present in the target receptacle
- Count how many target objects have been successfully placed
- Note: Other items in the receptacle do not invalidate success

### 3. Make a Verification Decision

| Condition | Decision | Output |
|-----------|----------|--------|
| All required objects confirmed in target receptacle | Complete | `Verification: Task complete. No further action needed.` |
| Some but not all objects placed | Incomplete | `Verification: Task incomplete. Continue searching for {missing object(s)}.` |
| Observation says "Nothing happened" | Ineffective | `Verification: Last action was ineffective. Re-assess and try a different approach.` |
| Object placed in wrong receptacle | Incorrect | `Verification: Object placed in wrong location. Retrieve and redirect to {correct receptacle}.` |

### 4. Output Format
Output **only** the verification decision. Do not output the next action. This skill informs the planning for the next step, not the execution.

## Example

**Task:** "Find two pen and put them in garbagecan."

```
Action: put pen 3 in/on garbagecan 1
Observation: You put the pen 3 in/on the garbagecan 1.

> Verification: Task incomplete. Continue searching for pen (1 of 2 placed).

Action: put pen 1 in/on garbagecan 1
Observation: You put the pen 1 in/on the garbagecan 1.

> Verification: Task complete. No further action needed.
```

**Example — Failed action:**

```
Action: put pen 3 in/on garbagecan 1
Observation: Nothing happened.

> Verification: Last action was ineffective. Re-assess and try a different approach.
```

## Error Handling
- **Ambiguous observation**: If the observation does not clearly confirm or deny placement, navigate to the target receptacle and re-examine it to get an updated state.
- **Quantity tracking**: For multi-object tasks, maintain a running count. Re-examine the target receptacle if the count is uncertain.
