---
name: webshop-initial-search
description: Performs the first search on an e-commerce platform using keywords derived from parsed user requirements. Use when starting product discovery from a user instruction or when previous search results are insufficient and a new query is needed. It formulates a search query from the criteria (e.g., '24 pack of 7.5 ounce bottles of non-gmo classic tonic') and executes the search[] action, returning the initial result page.
---
# Instructions
1.  **Parse Requirements:** Extract the core product criteria from the user's instruction. Focus on:
    *   **Quantity & Packaging:** (e.g., "24 pack")
    *   **Size:** (e.g., "7.5 ounce bottles")
    *   **Key Attributes:** (e.g., "non-gmo", "classic tonic")
    *   **Price Constraint:** (e.g., "price lower than 50.00 dollars") - This is for validation later, not for the initial search query.

2.  **Formulate Search Query:** Combine the extracted criteria into a concise, natural-language search string. Prioritize specificity to filter results effectively.
    *   **Example:** For "i would like to get a 24 pack of 7.5 ounce bottles of non-gmo classic tonic", the query should be: `24 pack of 7.5 ounce bottles of non-gmo classic tonic`
    *   **Do not** include the price constraint in the search keywords.

3.  **Execute Search:** Use the platform's `search[<keywords>]` action with your formulated query.
    *   The action output will be the initial search results page observation.

4.  **Next Steps:** After executing this skill, proceed to evaluate the search results. The next actions will typically involve clicking on a promising product listing or refining the search.

## Example

**User instruction:** "I would like to find a 24 pack of 7.5 ounce bottles of non-gmo classic tonic, and target price lower than 50.00 dollars"

**Thought:** I need to search for the product using the key attributes. I will omit the price constraint from the search keywords.

**Action:** `search[24 pack of 7.5 ounce bottles of non-gmo classic tonic]`

**Observation:** The platform returns the initial search results page with matching product listings.
