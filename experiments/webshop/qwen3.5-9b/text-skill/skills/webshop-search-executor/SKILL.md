---
name: webshop-search-executor
description: >-
  Executes a search on an e-commerce platform using parsed keywords. Use when you have
  extracted search terms from a user query and need to perform the actual search[] action
  on the WebShop interface. Takes structured search terms and performs a search action,
  returning a list of product results for evaluation.
---
# Instructions

## 1. Trigger Condition
Activate when the user provides an instruction to find or buy a product with specific attributes. The instruction will be in the `Observation`.

## Edge Cases
*   If the search returns 0 results, retry with a broader query (e.g., drop the least critical attribute).
*   If all results exceed the user's price constraint, try adding price-related terms or narrowing the product category.

## 2. Action Protocol
You must respond using **only** the following two action formats. Your final response must be in the exact structure:

**Thought:** [Your reasoning about the search terms to use based on the user's requirements.]
**Action:** `search[<keywords>]`

Or, if selecting a product from results:

**Thought:** [Your reasoning about which product best matches.]
**Action:** `click[<product_id>]`

## 3. Example

**Instruction:** "i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars"

**Observation:** WebShop search page is displayed with a search bar.

**Thought:** The user needs a clip-in hair extension that is long and natural looking, under $40. I will search for the core product type with key attributes to get relevant results.
**Action:** `search[long natural looking clip-in hair extension]`

---

**Instruction:** "i want a pack of organic green tea bags, price lower than 10.00 dollars"

**Observation:** WebShop search page is displayed with a search bar.

**Thought:** The user wants organic green tea bags under $10. I will search with the key product terms.
**Action:** `search[organic green tea bags]`
