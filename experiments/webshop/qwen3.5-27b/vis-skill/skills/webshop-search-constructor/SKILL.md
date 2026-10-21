---
name: webshop-search-constructor
description: This skill formulates an effective search query from structured product criteria. It is used when initiating a product search on an e-commerce platform, converting attributes into a concise, relevant keyword string. The input is a set of parsed criteria (e.g., 'jewelry box', '10 slots', 'easy to clean', 'price < $60'), and the output is a formatted search action string (e.g., 'search[jewelry box, 10 slots, less than $60, easy to clean]') optimized for the platform's search engine.
---
# Instructions
When tasked with finding a product on an e-commerce platform, use this skill to construct the initial search query.

## Core Process
1.  **Receive Criteria:** You will be given a set of structured product criteria (e.g., product type, attributes, price limit).
2.  **Construct Query:** Use the `construct_search_query.py` script to transform the criteria into a single, optimized search string. The script handles formatting and keyword ordering.
3.  **Execute Action:** Format the script's output into the required platform action: `search[<constructed_query>]`.

## Workflow Example
**Input Criteria:** `{'product': 'jewelry box', 'attributes': ['10 slots', 'easy to clean'], 'price_limit': 60.00}`
**Skill Execution:**
1.  Run the script: `python scripts/construct_search_query.py --product "jewelry box" --attributes "10 slots" "easy to clean" --price-limit 60.00`
2.  The script returns: `jewelry box, 10 slots, less than $60, easy to clean`
3.  You output the final action: `search[jewelry box, 10 slots, less than $60, easy to clean]`

## Important Notes
*   **Use the Script:** Always delegate query construction to the bundled script. Do not attempt to manually format the search string.
*   **Input Format:** Ensure criteria are passed to the script as discrete arguments (product, list of attributes, price limit). Parse the user's instruction into this structure before calling the script.
*   **Output Format:** The final action must strictly follow the `search[<query>]` syntax.
