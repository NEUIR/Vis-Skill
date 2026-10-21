---
name: webshop-result-analyzer
description: This skill evaluates a list of search results against the target product criteria. Trigger it when a search result page is observed, to identify promising product listings. It analyzes each result's title, price, and brief description to shortlist items that best match the required attributes (like size, color, and price), outputting a prioritized list of candidate product IDs or links for further inspection.
---
# Instructions

Trigger this skill when you observe a search result page (e.g., containing "Page 1 (Total results: 50)" and multiple product listings).

## 1. Extract User Requirements
First, parse the user's instruction from the observation. Identify the following key attributes:
- **Product Type:** (e.g., "woman's us size 5 high heel shoe")
- **Specific Attributes:** (e.g., "rubber sole", "color patent-beige")
- **Price Constraint:** (e.g., "price lower than 90.00 dollars")

## 2. Analyze Search Results
For each product listing in the observation (typically formatted as `[ASIN/Product ID] [SEP] [Title] [SEP] [Price Range]`):
1.  Extract the **Product ID** (e.g., B09GXNYJCD).
2.  Extract the **Product Title**.
3.  Extract the **Price**. Convert any range (e.g., "$49.99 to $54.99") to its maximum value for comparison against the budget.
4.  Perform a **textual match** between the title/description and the required attributes (size, color, material like "rubber", product type).

## 3. Score and Prioritize
Use the bundled Python script `analyze_results.py` to perform a consistent, deterministic analysis.
1.  **Run the script** with the extracted user requirements and the list of product data.
2.  The script will output a **prioritized list** of candidate Product IDs, sorted by a match score.

## 4. Output and Next Action
Present the analysis in this format:
