---
name: webshop-product-search
description: >-
  Performs an initial product search using a web interface by generating appropriate search
  keywords based on interpreted query criteria. Use when starting a product discovery task,
  when returning to search results after rejecting a product, or when the observation contains
  a search interface prompt. The skill inputs structured search parameters and outputs a list
  of candidate products from the search results page.
---
# Skill: WebShop Product Search

## Core Workflow
1.  **Interpret Criteria:** Extract key product attributes from the user's instruction (e.g., product type, features, price constraints).
2.  **Formulate Query:** Construct a concise search string prioritizing the core product type and critical attributes. Start broad; refine if results are poor.
3.  **Execute Search:** Use the `search[keywords]` action to submit the query. Keywords must be a single string enclosed in brackets.
4.  **Validate Results:** Success is marked by a "Page 1" observation containing product listings. If 0 results are returned, re-invoke with a broader or alternative query. If all results exceed the user's price constraint, refine with additional terms.

## Key Decision Logic
*   **Trigger Condition:** This skill is the entry point for a new product discovery task. Also activate when the observation contains "[SEP] Search" or "[SEP] Back to Search".
*   **Output Handoff:** The search results page is passed to downstream skills (e.g., `webshop-result-filter`, `webshop-product-evaluator`) for individual product analysis.

## Constraints
*   Do not use this skill if no search action is available in the current state.
*   Do not include multiple actions or extra formatting in the search keywords.

## Example from Trajectory
**Instruction:** "i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars"
**Skill Execution:**
1.  Criteria identified: `product_type="clip-in hair extension"`, `attribute="long"`, `attribute="natural looking"`, `budget="<40.00"`.
2.  Query formulated: Prioritizes core product type and key attribute -> `"long clip-in hair extension"`.
3.  Action: `search[long clip-in hair extension]`
