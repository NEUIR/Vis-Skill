---
name: scienceworld-liquid-pourer
description: Transfers the contents of a source liquid container into a target container for mixing or preparation. Use this skill when you need to combine multiple substances (such as paints or chemicals) into a single vessel, or when a liquid must be moved from one container to another before processing.
---
# Skill: Liquid Pourer

## Purpose
Transfer a liquid from a source container to a target container, typically as a preparatory step for mixing or chemical combination.

## When to Use
- Combining multiple liquids or substances into one vessel for mixing.
- Moving a liquid to a more suitable container before heating or processing.
- Preparing ingredients for a recipe that requires pouring.

## Core Workflow
1. **Identify Containers:** Use `look around` to locate the source container (with the liquid) and the target container (destination vessel).
2. **Perform Transfer:** `pour OBJ into OBJ` with precise object identifiers.
3. **Verify:** `look at OBJ` on the target container to confirm the transfer succeeded.

## Key Actions
| Action | Purpose |
|--------|---------|
| `look around` | Locate source and target containers |
| `pour OBJ into OBJ` | Transfer liquid contents |
| `look at OBJ` | Verify transfer success |
| `mix OBJ` | Combine contents (separate skill, used after pouring) |

## Example
**Task:** Create orange paint by mixing red and yellow paint.

1. `look around` — find `wood cup (containing red paint)`, `wood cup (containing yellow paint)`, and `jug`
2. `pour wood cup (containing red paint) into jug`
3. `pour wood cup (containing yellow paint) into jug`
4. `mix jug` — produces orange paint

## Important Notes
* Object identifiers must be precise (e.g., `wood cup (containing red paint)`). Use `look around` or `examine` to confirm exact names.
* Ensure the target container can receive the substance without contamination.
* Pouring only transfers contents — use `mix` as a separate step to combine them.
