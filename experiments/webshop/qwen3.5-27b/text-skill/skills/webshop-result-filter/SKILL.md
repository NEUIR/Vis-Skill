---
name: webshop-result-filter
description: >-
  Filters search results by evaluating product listings against specific user constraints
  like price, features, or ratings. Use when you are on a search results page and need to
  systematically identify which products meet all given criteria before selecting one for
  closer inspection. Takes a list of products with their details and outputs a filtered
  subset that meets the defined requirements.
---
# Skill: webshop-result-filter

## When to Use
Activate this skill when you are on a search results page in a web shopping environment and need to systematically evaluate which products meet a user's specific, multi-faceted requirements (e.g., "price lower than 40.00 dollars" AND "natural looking").

## Core Instruction
1.  **Parse the Instruction:** Extract the user's constraints from the instruction. Common constraints include:
    *   **Price:** A maximum or target price (e.g., `price lower than 40.00 dollars`).
    *   **Features:** Specific attributes or keywords (e.g., `natural looking`, `long`, `clip-in`).
    *   **Ratings:** A minimum rating threshold (if available in the observation).

2.  **Validate Constraints:** Confirm you have at least one constraint extracted before proceeding. If the instruction contains no filterable criteria, skip filtering and select the first available product.

3.  **Parse the Observation:** Extract the list of products from the search results page (Product ID, Title, Price, Rating if available).

4.  **Apply Filters:** For each product, check it against **all** extracted constraints:
    *   **Price Filter:** Is the product price strictly below the user's maximum?
    *   **Keyword/Feature Filter:** Does the title/description contain the required feature keywords?
    *   **Rating Filter:** If a rating constraint exists, does the product meet the minimum?

5.  **Output Decision:** Select the **first product** that passes all criteria as the primary candidate for `click[product_id]`. If no product passes:
    *   Try `click[Next >]` to check additional result pages.
    *   If no more pages, use `search[refined keywords]` with adjusted terms.

## Example from Trajectory
*   **User Instruction:** `i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars`
*   **Extracted Constraints:**
    *   `price < 40.00`
    *   Keywords: `natural` (implied from "natural looking")
*   **Observation (First Page):** Contains ~10 products with IDs, titles, and prices.
*   **Filtering Process:**
    1.  `B09C337K8S`: Price $29.99 (< $40.00). Title contains "Natural Looking". **PASSES**.
    2.  `B093BKWHFK`: Price $63.99 (> $40.00). **FAILS** on price.
    3.  `B099K9Z9L2`: Price $43.99 (> $40.00). **FAILS** on price.
*   **Result:** `B09C337K8S` is selected as the top matching candidate.

## Next Action
After identifying a matching product, click on it for more details:

**Thought:** [Summarize which constraints each product passed or failed, justify your selection.]
**Action:** `click[<matching_product_id>]`
