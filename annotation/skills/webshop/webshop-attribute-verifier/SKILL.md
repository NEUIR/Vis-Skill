---
name: webshop-attribute-verifier
description: >
  Verifies product attributes on a web shop detail page by extracting price, comparing color availability,
  validating specifications, and confirming option selections against user requirements before purchase.
  Use when you need to check if a product matches requirements, verify product details before buying,
  confirm item specifications on an online store product page, or validate that price, color, size,
  or other attributes satisfy the user's constraints. Outputs a `Thought:` assessment followed by a
  `click[value]` action to select the matching option and proceed, or navigates back to search if
  the product does not match.
---

# Instructions

You are verifying product attributes on a detailed product page. Your goal is to confirm the product matches the user's requirements before proceeding.

## Input

You receive an **Observation** containing:
- The original user instruction (e.g., "find a black leather wallet, price lower than 40.00 dollars")
- Product details: Title, Price, Rating
- Available options (e.g., color buttons like "black", "brown")
- Available actions (e.g., "Buy Now", "Back to Search", "Description")

## Core Workflow

1. **Parse Requirements:** Extract every target attribute from the user instruction — color, material, size, max price, and any other constraints.
2. **Inspect Page:** Read the observation for the product's displayed price, title, available option buttons, and specifications.
3. **Compare Price:** Confirm the product's price is **strictly lower than** the user's maximum price. If not, navigate back immediately.
4. **Verify Attributes:** Check each remaining requirement (color, material, size, etc.) against the product's details and available options.
5. **Select or Reject:**
   - If all requirements are satisfied, `click[option]` on the required attribute (e.g., `click[black]`) **first**, then `click[Buy Now]`.
   - If any requirement fails, `click[Back to Search]` to find a different product.

## Action Format

- `click[value]` — interact with a page element. The `value` must exactly match a clickable item from the observation.
- `search[keywords]` — start a new search only if the current product cannot meet requirements.
- Every response must begin with `Thought:` stating which requirement is being checked and the verdict.

## Critical Rules

- **Price is strict:** The price must be strictly less than the user's budget — equal is not acceptable.
- **Select before buying:** Always click the required attribute option (e.g., `click[black]`) before clicking `click[Buy Now]`. Never skip this step.
- **One action per turn:** Output exactly one action after your `Thought:`.

## Example

**User instruction:** "Find a black leather wallet with price lower than 40.00 dollars"

**Observation:**
```
[Back to Search] [< Prev]
Title: Genuine Leather Bifold Wallet - Multiple Colors
Price: $35.99
Rating: ★★★★☆
Options: [black] [brown] [tan]
Actions: [Buy Now] [Description] [Back to Search]
```

**Response:**
```
Thought: The user wants a black leather wallet under $40. The price is $35.99, which is strictly less than $40 — price check passes. The title mentions "Leather" which matches the material requirement. The color "black" is available as a selectable option. All requirements are satisfied. I will select the black option first.
Action: click[black]
```

**Next turn — after selecting black:**
```
Thought: The black option is now selected. All requirements verified. Proceeding to purchase.
Action: click[Buy Now]
```
