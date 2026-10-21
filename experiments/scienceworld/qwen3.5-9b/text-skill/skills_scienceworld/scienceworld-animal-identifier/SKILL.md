---
name: scienceworld-animal-identifier
description: Use when the agent needs to locate, identify, and focus on a specific animal or biological entity in the ScienceWorld environment. This skill handles tasks involving animal comparison, examination, or interaction (such as determining lifespan extremes) by navigating to the correct location with "teleport to", surveying with "look around", and executing "focus on ANIMAL" with the exact entity name.
---
# Skill: scienceworld-animal-identifier

## Purpose

Locate and focus on a specified animal or biological entity within the ScienceWorld environment for tasks involving animal comparison, examination, or interaction (e.g., determining lifespan extremes, identifying species).

## When to Use

- When a task requires finding and focusing on a specific animal or biological entity
- When comparing attributes (lifespan, size, type) across multiple animals
- When the task instructs examining or interacting with an animal in a specific location

## Workflow

1. **Navigate to the target location** -- If the animal is in a specific room (e.g., "outside", "greenhouse"), execute `teleport to <LOCATION>`.
2. **Survey the area** -- Use `look around` to list all visible objects and entities.
3. **Identify the target** -- Parse the observation to find the exact animal name as displayed (e.g., `baby dragonfly`, `parrot egg`, `adult blue jay`).
4. **Execute focus** -- Run `focus on <ANIMAL>` using the exact name from the observation.
5. **Confirm success** -- Wait for the system response: `"You focus on the <ANIMAL>."`.

## Examples

### Example 1: Find the animal with the longest lifespan

**Task:** "Find the animal with the longest life span and focus on it."

```
> teleport to outside
You teleport to outside.

> look around
You see: baby dragonfly, adult blue jay, parrot egg, oak tree, grass.

> focus on adult blue jay
You focus on the adult blue jay.
```

(Blue jays have a longer lifespan than dragonflies among the visible animals.)

### Example 2: Sequential identification of two animals

**Task:** "First focus on the longest-lived animal, then the shortest-lived."

```
> teleport to outside
You teleport to outside.

> look around
You see: baby dragonfly, adult blue jay, parrot egg.

> focus on adult blue jay
You focus on the adult blue jay.

> focus on baby dragonfly
You focus on the baby dragonfly.
```

## Key Principles

- **Exact names** -- The `focus on` action requires the precise object name as it appears in `look around` (e.g., `"baby dragonfly"`, not just `"dragonfly"`).
- **Navigate first** -- Always verify your location and `teleport to` the correct room before surveying.
- **Sequential execution** -- When focusing on multiple animals in sequence, complete each `focus on` action before proceeding to the next.
