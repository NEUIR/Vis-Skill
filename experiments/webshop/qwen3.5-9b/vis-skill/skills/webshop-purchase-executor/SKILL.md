---
name: webshop-purchase-executor
description: Executes the purchase action for a confirmed suitable product on an e-commerce page. Use when a product has been verified against the user's requirements (price, attributes) and the decision is to buy. Performs a click[Buy Now] action on the purchase button to complete the transaction.
---
# Skill: webshop-purchase-executor

## Purpose
Execute the final purchase transaction for a verified product on an e-commerce platform. This skill is the terminal action in a shopping workflow, initiated only after a product has been confirmed to meet the user's criteria (e.g., price, description).

## When to Use
- **Trigger:** You are on a product detail page.
- **Prerequisite:** The product has been verified against the user's requirements (e.g., price < $40.00, "natural looking").
- **User Intent:** The user has made a decision to buy the product.
- **Available Action:** A purchase button (e.g., "Buy Now") is present in the list of clickable elements.

## Core Action
Perform a `click` action on the purchase button.

## Execution Logic
1.  **Locate Purchase Button:** Identify the purchase button from the available actions. Common identifiers include:
    - `buy now`
    - `add to cart` (if the instruction implies immediate purchase)
    - `purchase`
2.  **Validate Context:** Ensure you are on a product page and the prior verification step is logically complete.
3.  **Execute Click:** Perform the `click[purchase_button_value]` action using the exact value from the available actions list.

## Important Notes
- Do not use this skill for browsing, searching, or product comparison.
- The skill assumes price and specification validation occurred in a previous step.
- If multiple purchase options exist (e.g., "Buy Now", "Add to Cart & Checkout"), prefer the most direct path to complete the transaction, typically "Buy Now".

## Example

**Scenario:** The user wants a natural-looking wig priced under $40.00. You are on a product detail page for "Natural Looking Human Hair Wig - $34.99" and have verified it meets all criteria.

**Thought:** The product matches the user's requirements (natural looking, price $34.99 < $40.00). I should proceed with the purchase.

**Action:** `click[Buy Now]`

**Observation:** The purchase is confirmed and the transaction is complete.
