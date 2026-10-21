---
name: webshop-product-detail-navigator
description: Navigates to and interacts with a specific product's detail page. This skill is triggered after a candidate product ID is selected from search results. It performs the click action to load the product page and then identifies available interactive elements (like flavor or size selectors) that need to be configured to match the user's requirements.
---
# Instructions

## Purpose
This skill is used to navigate from a product search result to the product's detail page and configure it according to the user's specific requirements (e.g., flavor, size, dietary needs).

## When to Use
Use this skill when you have identified a candidate product from a search result list and need to:
1.  Load its detailed product page.
2.  Verify its attributes against the user's instruction.
3.  Select the correct options (like `flavor` or `size`) to match the user's request.
4.  Proceed to the final purchase step if all criteria are met.

## Core Workflow
1.  **Navigate to Product Page:** Click on the product ID (e.g., `B06Y96MXJV`) from the search results to load its detail page.
2.  **Parse the Page State:** Once on the product page, the observation will contain the product title, price, and a list of available interactive elements (clickables). Analyze these to find configuration options.
3.  **Match Requirements:** Cross-reference the user's original instruction with the available options. Look for keywords related to:
    *   **Product Type/Flavor:** (e.g., "smoked peppered bacon", "smoked bacon").
    *   **Attributes:** (e.g., "gluten free", "vegetarian").
    *   **Size/Packaging:** (e.g., "4 ounce (pack of 2)").
    *   **Price:** Ensure the displayed price is below any specified maximum.
4.  **Configure the Product:** Sequentially click on the option values that match the user's requirements. The typical order is to select the correct `flavor` first, then the correct `size`.
5.  **Finalize:** After configuring the product, if all requirements are satisfied, click the final action button (e.g., `Buy Now`).

## Action Format
Your actions must strictly follow the required format:
*   `click[<exact_value>]` - The `<exact_value>` must match one of the strings in the "clickables" list from the observation, including case and spacing.
*   `search[<keywords>]` - Use only if you need to go back to search. This skill focuses on post-search navigation.

## Key Logic (Handled by Script)
The deterministic sequence of checking for and selecting flavor and size options is handled by the bundled script `configure_product.py`. Use it to generate the correct click actions after loading the product page.

## Important Notes
*   **Validation:** Do not click an option if it does not appear in the current list of clickables.
*   **Observation:** Always read the full observation after each action. The list of clickables updates after each selection.
*   **Goal:** The final goal is to have a product configured per the user's instruction and ready for purchase.
