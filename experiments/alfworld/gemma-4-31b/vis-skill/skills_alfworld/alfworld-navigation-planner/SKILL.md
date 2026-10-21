---
name: alfworld-navigation-planner
description: Plans a path to move the agent between receptacles to search for target objects. Use this when you need to traverse the environment to reach a specific location or systematically explore multiple areas. It takes the current location and destination receptacle as input, and outputs the next 'go to' action to approach the target.
---
# Instructions

Use this skill to navigate between receptacles (e.g., bed, desk, drawer) in a household environment to locate objects. The core logic is handled by the bundled script.

## Input/Output Format
- **Input:** Provide the agent's **current location** and the **target receptacle**.
- **Output:** The skill will return the next `go to {recep}` action to execute.

## How to Use
1.  **Identify your goal.** Determine which receptacle you need to search (e.g., `desk 1` to find a `desklamp`).
2.  **Call the skill.** Pass your current location and the target receptacle.
3.  **Execute the action.** Perform the `go to` action returned by the skill.
4.  **Observe and repeat.** After moving, observe the new location. If the target object is not found, use this skill again to plan movement to the next most promising receptacle.

## Key Principles
- **Systematic Search:** Move efficiently between likely receptacles instead of random exploration.
- **Adaptive Planning:** If an action fails ("Nothing happened"), the script's logic helps choose an alternative path.
- **Context Preservation:** Always note your current location after each move for the next planning step.

**Example Thought Process:**
> Thought: I need to find a desklamp. The `desk 1` is a likely location. I am currently at `bed 1`.
> Action: Use `alfworld-navigation-planner` with current location `bed 1` and target `desk 1`.
> Result: `go to desk 1`
