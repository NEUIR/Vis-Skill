---
name: webshop-product-detail-check
description: >-
  Examines a specific product's detailed page to verify it matches the user's requirements,
  checking price, description, features, and reviews. Use when you have navigated to a
  product detail page from search results and need to confirm the product meets all
  user-specified constraints before purchasing. It provides a final suitability assessment
  with a clear proceed-or-reject recommendation.
---
# Skill: Product Detail Verification

## Trigger
Activate this skill when the agent has navigated to a specific product's detailed page from a search results list. The agent should have a user query containing specific requirements (e.g., product type, attributes, price constraints).

## Core Objective
Analyze the product detail page to confirm if the product meets all user-specified constraints. Provide a clear "Proceed" or "Reject" recommendation.

## Execution Workflow

1.  **Parse User Requirements:** Extract key constraints from the user's instruction. Primary constraints are typically **Price** and **Key Attributes** (e.g., "long", "clip-in", "natural looking"). Store these in a structured format.

2.  **Scrape Product Page Data:** Systematically extract the following information from the current page observation:
    *   **Product Title**
    *   **Price**
    *   **Key Description Snippets** (from Description/Features sections)
    *   **Rating** (if available)
    *   **Availability of "Buy Now"** action.

3.  **Constraint Verification:** Compare the scraped data against the user's requirements.
    *   **Price Check:** Is the product price strictly less than the user's maximum? (e.g., < $40.00).
    *   **Attribute Check:** Do the title and description contain terms matching the user's key attributes? (e.g., confirm "clip-in" and "natural looking" are present).
    *   **Suitability Flag:** Product is suitable only if **ALL** hard constraints (price, mandatory attributes) are satisfied.

4.  **Decision & Action:**
    *   **If SUITABLE:** Conclude the product matches requirements. The recommended action is `click[buy now]`.
    *   **If NOT SUITABLE:** Conclude the product fails one or more constraints. The recommended action is to return to search (`click[back to search]`) to find another candidate.

## Output Format
Your response must use the following format:

**Thought:** [Your reasoning, summarizing the user's constraints, the product details found, and the result of the verification check.]
**Action:** [The single, valid action to take: `click[buy now]` or `click[back to search]`]

## Example

**Instruction:** "i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars"

**Observation:** Product page for "22 Inch Long Natural Looking Clip-In Hair Extension" — Price: $34.99, Description mentions "clip-in", "natural looking", "synthetic fiber".

**Thought:** The user needs a long, natural-looking, clip-in hair extension under $40. This product is priced at $34.99 (< $40.00), the title confirms "long" and "natural looking", and the description confirms "clip-in". All constraints are satisfied.
**Action:** `click[buy now]`

---

**Observation:** Product page for "Premium Human Hair Extension" — Price: $45.99, Description mentions "clip-in", "natural".

**Thought:** The user's budget is under $40. This product is priced at $45.99, which exceeds the $40 limit. Price constraint fails.
**Action:** `click[back to search]`

## Notes
*   Prioritize information explicitly on the product page. Assume missing information (like a Rating of "N.A.") does not disqualify a product unless specified by the user.
*   The "Buy Now" button must be present in the available actions list for the purchase action to be valid.
