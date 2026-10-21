---
name: webshop-attribute-selector
description: Selects specific product attributes (e.g., flavor, size) on a product detail page to match user requirements. Triggered when a product page presents configurable options and the parsed query specifies needed values. It identifies the correct option from available clickables (e.g., 'smoked bacon' for flavor, '4 ounce (pack of 2)' for size) and executes the selection click.
---
# Instructions

## When to Use
Activate this skill when ALL of the following are true:
1.  You are on a product detail page (not a search results page).
2.  The observation contains configurable product options (e.g., "flavor", "size") listed as available actions.
3.  The user's instruction specifies required values for one or more of these attributes (e.g., "gluten free", "4 ounce (pack of 2)", "smoked peppered bacon").

## Core Process
1.  **Parse Requirements:** From the user's instruction, extract the required attribute values (e.g., target flavor, size, dietary feature).
2.  **Map to Options:** Examine the list of available actions (clickables). Identify which clickable text corresponds to each required attribute value.
    *   Use exact or fuzzy matching (see `scripts/match_attribute.py` for logic).
    *   Prioritize exact matches. For dietary features like "gluten free" or "vegetarian," the product title or description may already confirm this; your goal is to select the correct flavor/size.
3.  **Execute Selection:** For each identified attribute option, perform a `click[value]` action, where `value` is the exact text of the matching clickable.
4.  **Finalize:** After all required attributes are selected, if a "Buy Now" or similar action is available and the instruction implies a purchase, you may proceed to click it.

## Important Notes
*   Do not use this skill on search results pages. Use search actions there.
*   If a required attribute cannot be matched to any available clickable, do not click. Re-evaluate the instruction or consider a different product.
*   The observation may refresh after each click, showing the same page with selected options highlighted. Continue until all specified attributes are satisfied.
