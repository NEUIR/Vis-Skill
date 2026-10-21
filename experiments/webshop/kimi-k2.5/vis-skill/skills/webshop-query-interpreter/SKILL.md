---
name: webshop-query-interpreter
description: >-
  Interprets a user's shopping query to extract key product requirements such as item type,
  attributes, and constraints. Use when a new shopping instruction is received and you need
  to parse natural language into structured criteria (product category, desired features,
  price limits) before performing a search. The output is a clear set of search parameters
  for use in subsequent product lookup steps.
---
# Instructions

## Trigger
Activate this skill when a new shopping instruction is received from the user. The instruction will be a natural language request for a product.

## Core Process

1.  **Parse the Instruction:** Analyze the user's query to identify the following key components:
    *   **Primary Product Type:** The main item being requested (e.g., "hair extension").
    *   **Key Attributes/Features:** Descriptive requirements (e.g., "long", "clip-in", "natural looking").
    *   **Constraints:** Hard limits, especially price (e.g., "price lower than 40.00 dollars").

2.  **Structure the Output:** Format the extracted information into a clear, structured set of search parameters. Use the following template:

    ```
    Product Type: <primary item>
    Attributes: <comma-separated features>
    Constraints: <price limits or other hard requirements>
    Search Query: <recommended search string>
    ```

3.  **Hand Off:** Pass the structured parameters to the search execution skill (e.g., `webshop-search-executor` or `webshop-product-search`) to perform the actual query.

## Example

**Instruction:** "i need a long clip-in hair extension which is natural looking, and price lower than 40.00 dollars"

**Thought:** The user wants a hair extension. Key attributes are "long", "clip-in", and "natural looking". The hard constraint is price under $40.00.

**Structured Output:**
```
Product Type: clip-in hair extension
Attributes: long, natural looking
Constraints: price < $40.00
Search Query: long clip-in natural looking hair extension
```

---

**Instruction:** "i want a pack of 6 moisturizing body wash bars with shea butter, price less than 20 dollars"

**Thought:** The user wants body wash bars. Key attributes are "moisturizing", "shea butter", quantity "6 pack". Hard constraint is price under $20.00.

**Structured Output:**
```
Product Type: body wash bars
Attributes: moisturizing, shea butter, 6 pack
Constraints: price < $20.00
Search Query: moisturizing body wash bars shea butter 6 pack
```
    