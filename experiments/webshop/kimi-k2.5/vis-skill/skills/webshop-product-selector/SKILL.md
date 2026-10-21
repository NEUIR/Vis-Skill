---
name: webshop-product-selector
description: >-
  Selects the most promising product candidate from a filtered shortlist for detailed
  inspection. Use when you are on a search results page with multiple viable items and need
  to make a final selection based on a heuristic such as best feature match or lowest price.
  Triggers navigation to the selected product's detail page via click[product_id].
---
# Instructions
Use this skill when you are on a search results page with multiple product options that appear to meet the user's core requirements.

## Decision Process
1.  **Assess the List:** Review the provided observation, which contains a list of products. Each product entry typically includes a Product ID (e.g., `B093C1THVP`), a title, and a price.
2.  **Apply Selection Heuristic:** Use the bundled `selector.py` script to programmatically identify the best candidate. The script implements the following logic:
    *   Filters products strictly under the user's specified maximum price.
    *   Scores remaining products based on keyword matching in the title against the user's core requirements (e.g., "portable", "easy to carry", "double horn").
    *   Selects the product with the highest score. In case of a tie, it chooses the lowest-priced item.
3.  **Execute Action:** The final action is always to click on the selected product's ID to navigate to its detail page for final verification and purchase. The action format is `click[product_id]`.

## Thought Process Template
Structure your reasoning before acting using this template:

**Thought:** [List the user's requirements, evaluate each candidate product against them, explain why the selected product is the best match.]
**Action:** `click[<product_id>]`

## Example

**Instruction:** "i need a portable double horn bluetooth speaker that is easy to carry, and price lower than 50.00 dollars"

**Observation (Search Results):**
- B093C1THVP: "Portable Double Horn Bluetooth Speaker — Compact & Easy to Carry" — $32.99
- B08XYZ1234: "Double Horn Wireless Speaker — Premium Sound" — $55.00
- B07ABC5678: "Mini Portable Bluetooth Speaker" — $19.99

**Thought:** The user needs a portable, double horn bluetooth speaker that is easy to carry, under $50. B093C1THVP at $32.99 matches "portable", "double horn", and "easy to carry" — all keywords present in the title. B08XYZ1234 exceeds the $50 budget. B07ABC5678 is within budget but lacks "double horn". Selecting B093C1THVP as the best match.
**Action:** `click[B093C1THVP]`
