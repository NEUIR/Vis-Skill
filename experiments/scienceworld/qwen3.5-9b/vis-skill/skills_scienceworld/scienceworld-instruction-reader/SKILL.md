---
name: scienceworld-instruction-reader
description: Reads a recipe or note from the inventory using the 'read' action and extracts key information. Use this skill when you have acquired a recipe, note, or readable document in your inventory and need to extract explicit instructions, ingredient lists, or procedural steps before executing a task.
---
# Skill: Instruction Reader

## Purpose
Read and parse a document (recipe, note, or instructions) from your inventory to extract the information needed to execute the current task.

## When to Use
- After picking up a recipe, note, or readable document.
- When the task requires following written instructions (e.g., mixing chemicals, assembling components).
- Before starting a multi-step procedure that depends on written directions.

## Core Workflow
1. **Read the Document:** `read OBJ` on the document in your inventory.
2. **Extract Key Information:**
   - **Goal:** The final product or objective (e.g., "make salt water").
   - **Required Components:** Items or substances needed (e.g., "sodium chloride, water").
   - **Procedural Steps:** Actions or combinations described (e.g., "mix sodium chloride, water").
3. **Plan Next Actions:** Use the extracted information to determine which skills and actions to invoke next.

## Example
**Task:** Follow a recipe to make salt water.

1. `pick up recipe` — acquire the document
2. `read recipe in inventory` — output: "To make salt water, mix sodium chloride and water in a glass jar."
3. **Extracted info:**
   - Goal: salt water
   - Components: sodium chloride, water
   - Procedure: mix in glass jar
4. Proceed to fetch sodium chloride, water, and glass jar, then mix.
