---
name: alfworld-storage-explorer
description: Systematically explores storage receptacles (drawers, cabinets, shelves) to find an appropriate placement location for an object. Use when the agent needs to store an item but the exact target receptacle is unknown or ambiguous. Opens, inspects, and closes candidate receptacles to assess suitability, then places the object in the best match.
---
# Skill: Storage Explorer

## When to Use
Activate this skill when:
1. You have an object that needs to be stored/placed.
2. The exact target receptacle is unspecified or ambiguous.
3. You need to validate if a potential receptacle is appropriate before placing the object.

## Core Strategy
1. **Prioritize Exploration**: When the target location is unclear, systematically check available storage receptacles.
2. **Inspect Before Placing**: Always open and examine a receptacle's contents before deciding to place your object there.
3. **Maintain Environment State**: Close receptacles after inspection unless you're placing the object inside.

## Execution Pattern
Follow this decision flow:

### Phase 1: Initial Assessment
- Identify all potential storage receptacles in the environment (drawers, cabinets, etc.)
- If you already have the target object, proceed to Phase 2
- If you don't have the object, acquire it first (this skill focuses on storage exploration)

### Phase 2: Systematic Exploration
For each candidate receptacle:
1. **Navigate** to the receptacle
2. **Open** it (if closed)
3. **Observe** its contents
4. **Evaluate** suitability based on:
   - Current contents (are similar objects stored here?)
   - Available space
   - Logical categorization
5. **Close** the receptacle (unless placing object there)

### Phase 3: Decision Making
- If you find a clearly appropriate receptacle (e.g., contains similar objects), place your object there
- If no receptacle is clearly appropriate, choose based on:
  - Empty receptacles first
  - Proximity to related objects/areas
  - Consistency with environment patterns

## Key Considerations
- **Efficiency**: Don't explore receptacles that are clearly inappropriate (e.g., refrigerator for non-food items)
- **Context Clues**: Look for environmental hints about storage organization
- **Task Requirements**: Consider if the object needs special conditions (clean, cooled, heated)

## Error Handling
- If "Nothing happened" when trying to open/close, the receptacle might be locked or broken
- If unable to place object in a receptacle, try alternative locations
- If all exploration fails, reconsider your understanding of "appropriate" storage

## Example

**Task:** "Put a clean sponge in drawer."

**Input:** Holding `sponge 1` (already cleaned). Target type: drawer.

**Sequence:**
1. `go to drawer 1` → Observation: "You are at drawer 1."
2. `open drawer 1` → Observation: "You open the drawer 1. The drawer 1 is open. You see nothing."
3. `put sponge 1 in/on drawer 1` → Observation: "You put the sponge 1 in/on the drawer 1."

**Output:** The sponge 1 has been stored in drawer 1. Task complete.

## Output Format
Maintain the standard action format:
