---
name: alfworld-receptacle-searcher
description: This skill searches for a specific receptacle (e.g., garbage can, cabinet) by systematically exploring the environment, checking multiple locations until found. Trigger when the target receptacle is not initially visible, often after scanning nearby areas. It outputs the receptacle's location and may involve navigating through multiple waypoints.
---
# Instructions

## When to Use
Use this skill when you need to locate a specific receptacle (e.g., `garbagecan 1`, `cabinet 3`) that is not immediately visible in your current observation. The skill is triggered after an initial scan of nearby areas fails to reveal the target.

## Core Procedure
1. **Identify Target**: Confirm the exact name of the receptacle you need to find (e.g., from the task: "put a cool lettuce in garbagecan" -> target is `garbagecan 1`).
2. **Initial Check**: Quickly examine the most common or nearby receptacles mentioned in the initial observation (e.g., countertops, tables, open cabinets).
3. **Systematic Search**: If not found, begin a systematic exploration:
   - Use the `go to {recep}` action to navigate to candidate locations.
   - Prioritize receptacles that are typically in the environment (cabinets, countertops, tables).
   - If a candidate is closed (e.g., "The cabinet 2 is closed."), you may skip opening it unless the target could logically be inside. The trajectory shows that the target receptacle (`garbagecan 1`) was not inside any closed cabinet.
4. **Persistence**: Continue the search until the target is found. The trajectory demonstrates checking 20 cabinets before finally locating the garbage can.
5. **Output**: Once found, note the receptacle's location and state (e.g., "On the garbagecan 1, you see nothing."). You are now ready to use this receptacle for the next task action (e.g., `put lettuce 1 in/on garbagecan 1`).

## Key Notes
- **Efficiency**: The search pattern in the trajectory was exhaustive (checking all cabinets 1-20). In practice, you might infer likely locations based on receptacle type (e.g., a garbage can is often standalone, not inside a cabinet).
- **Error Handling**: If an action returns "Nothing happened", the receptacle may not be accessible or may require a different approach (e.g., opening first).
- **Integration**: This skill outputs the found receptacle's identifier. It does not perform the final task action (e.g., putting an item in the receptacle); that is handled by the main task orchestration.

## Example from Trajectory
**Goal**: Find `garbagecan 1`.
**Process**:
1. Checked `countertop 1`, `diningtable 1`, `diningtable 2` (not found).
2. Systematically went to `cabinet 1` through `cabinet 20` (all closed, not found).
3. Finally went directly to `garbagecan 1` (found).
**Output**: Location of `garbagecan 1` confirmed.
