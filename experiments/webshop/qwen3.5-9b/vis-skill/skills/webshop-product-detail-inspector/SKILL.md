---
name: webshop-product-detail-inspector
description: This skill examines a product detail page to verify specific attributes or select options like size, color, or quantity. It is used after clicking on a product, to confirm that the item matches the user's needs (e.g., checking for '10 slots' option, price, material). The input is the product page observation, and the output is an action to select a required variant (e.g., click[10 slots]) or proceed if criteria are satisfied.
---
# Instructions

## Purpose
You are an agent navigating a web shop. Your current task is to inspect a product detail page. You have arrived here after clicking on a product from the search results.

## Input
You will receive an **Observation** containing the product page state. This includes:
*   The original user instruction (e.g., "i am looking for an easy to clean jewelry box with 10 slots, and price lower than 60.00 dollars").
*   Product details: Title, Price, Rating.
*   Available options (e.g., size variants like "3 slot", "10 slots").
*   Available actions (e.g., "Buy Now", "Description").

## Core Logic
1.  **Parse the Instruction:** Identify the user's key requirements from the instruction (e.g., specific attribute: "10 slots", price constraint: "< $60").
2.  **Examine the Page:** Check the product's displayed **Price** against the user's budget. Check the available options for the required attribute.
3.  **Decision & Action:**
    *   If the required attribute (e.g., "10 slots") is present as a selectable option, click on it.
    *   If the product does not have the required attribute or exceeds the price limit, you should go "Back to Search".
    *   If the product meets all criteria (correct attribute selected, price is valid), you may proceed (e.g., click "Buy Now").

## Output Format
Your response must be in this exact format:
