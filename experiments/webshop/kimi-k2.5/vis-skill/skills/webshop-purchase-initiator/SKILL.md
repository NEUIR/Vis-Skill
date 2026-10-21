---
name: webshop-purchase-initiator
description: >-
  Initiates the purchase process by clicking the buy button once a product and its options
  have been selected and verified against requirements. Use when you are on a product detail
  page, all selection criteria (price, features, options) are satisfied, and the user is
  ready to proceed to checkout. Confirms the final price meets budget constraints before
  executing click[buy now].
---
# Instructions

## When to Use
Activate this skill when:
1. A product has been selected from search results.
2. All required product options (e.g., size, color, quantity) have been chosen.
3. The final displayed price has been verified to be within the user's specified budget.
4. The user's instruction implicitly or explicitly indicates readiness to purchase (e.g., "I need to buy...", "I want this one").

## Core Action
Execute the `click[buy now]` action to proceed to checkout.

## Verification Logic (Before Clicking)
Before clicking "Buy Now", you **must** perform these checks:
1. **Budget Check**: Confirm the final displayed `Price:` is less than the maximum budget specified in the user's original instruction. If no budget is specified, skip this check.
2. **Option Selection Check**: Ensure the required product options (like 'size', 'color') show a selected state or that your previous action has selected the correct option. The observation should reflect the chosen option.
3. **Product Match Check**: Verify the product title and description align with the user's core requirements (e.g., "teeth whitening that also freshens breath").

If any check fails, do not use this skill. Instead, revert to selecting the correct option or searching for a different product.

## Example

**Instruction:** "i need a teeth whitening toothpaste that also freshens breath, and price lower than 15.00 dollars"

**Observation:** Product page for "Whitening Toothpaste with Breath Freshener" — Price: $12.99, Size: 4oz (selected), Flavor: Mint (selected). Available actions: click[buy now], click[back to search].

**Thought:** The user needs a teeth whitening toothpaste that freshens breath, under $15. The current product is priced at $12.99 (< $15.00), matches "whitening" and "breath freshener" in the title, and all required options (size, flavor) are selected. All checks pass — proceeding to purchase.
**Action:** `click[buy now]`

## Error Handling
- If the "Buy Now" button is not present in the list of clickable actions, the action will be invalid. Do not force the action.
- If the price changes after selecting options and exceeds the budget, abort the purchase and inform the user.
