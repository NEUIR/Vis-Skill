---
name: scienceworld-mixture-creator
description: This skill chemically mixes the contents of a container using the 'mix' action. Use when all required ingredients (e.g., sodium chloride and water) are present inside a container and the agent needs to combine them to produce a new substance. The skill performs the mixing operation and outputs the resulting product, completing the synthesis step of the task.
---
# Skill: Chemical Mixture Creator

## When to Use
All required ingredients are confirmed inside a single container and you need to combine them into a new substance.

## Procedure
1. `examine <CONTAINER>` — verify all required ingredients are present.
2. `mix <CONTAINER>` — the environment processes the chemistry and produces the result.
3. `examine <CONTAINER>` or `focus on <NEW_SUBSTANCE>` — confirm the synthesis succeeded.

Do not `mix` unless the container holds the exact ingredients required by the recipe.
## Example
**Task:** Create salt water by mixing sodium chloride and water.
1. `examine glass cup` — observe: "containing water, sodium chloride"
2. `mix glass cup`
3. Expected result: the environment produces "salt water" inside the glass cup.
4. `focus on salt water` to confirm the synthesis is complete.

For common chemical recipes (e.g., salt water = sodium chloride + water), see `references/common_recipes.md`.
