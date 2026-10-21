---
name: webshop-variant-chooser
description: This skill selects a product variant (e.g., size, quantity) from available options on a product detail page. It is triggered when a product page displays multiple choices, assessing the options (like '1.1 pound (pack of 1)' or 'pack of 2') and selecting one that aligns with user needs, often defaulting to a standard single unit if unspecified.
---
# Instructions

## When to Use
Use this skill when you are on a product detail page and the observation contains a list of variant options (e.g., under a "size" or "quantity" heading). The skill is triggered by the presence of multiple clickable options that represent different configurations of the same product.

## Core Decision Logic
1.  **Parse the Observation:** Identify the list of available variant options from the observation text. These are typically presented as clickable values after a label like "size".
2.  **Assess User Needs:** Review the original user instruction for any explicit or implicit preferences regarding variant (e.g., budget constraints, desired quantity). If no preference is stated, default logic applies.
3.  **Select and Act:** Choose the most appropriate variant and execute a `click[value]` action.
    *   **If user instruction specifies a constraint (e.g., "price lower than 40.00 dollars"):** Ensure the selected variant's price (if displayed) complies.
    *   **If no user preference is stated:** Prefer the standard, single-unit option (e.g., "1.1 pound (pack of 1)" over "pack of 2") as a sensible default.
    *   **If the choice is ambiguous:** You may need to click on a variant to see its updated price/details before finalizing the selection, as shown in the trajectory.

## Action Format
Your response must strictly follow this format:
