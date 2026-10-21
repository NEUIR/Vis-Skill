def retrieve_relevant_skills_prompt(metadata, task):
    template = "You are an expert in task analysis and skills retrieval."
    prompt = f'''
Your task is to analyze the given task description and retrieve the most relevant skills from the provided metadata (name and description pairs).

# Principles
1.  **Contextual Understanding**: Thoroughly analyze the task description to grasp its requirements.
2.  **Relevance Matching**: Match the task requirements with the most relevant skills based on the metadata descriptions. Only include skills that are most helpful for accomplishing the task. (No more than 5 unless necessary)
3.  **Clarity and Precision**: Clearly list the names of the relevant skills in your output.
4.  **Output Format**: Return the skill names as a JSON list of strings. Must strictly match names in the provided metadata.
5.  **Allow Empty List**: If no relevant skills are found, return an empty list `[]`.

# Input Data
1.  **Metadata:** 
{metadata}
2.  **Task Description:**
{task}

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Relevant_Skill_Names> relevant skill names list in JSON format here </Relevant_Skill_Names>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]


def generate_overall_procedure_prompt(task, overall_procedure_examples, skill_contents):
    template = "You are a Senior Systems Architect."
    prompt = f'''
You are given a complex task description, relevant skill contents, and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as the "brain" for an AI Agent, telling it exactly how to solve the task.

# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Structure with Phases**: You MUST divide the task into logical phases (e.g., Phase 1: Search, Phase 2: Selection, Phase 3: Configuration, Phase 4: Purchase).
2.  **Use Exact Syntax**: When describing actions, you MUST quote the exact syntax from the "Available Actions" (e.g., use "search[keywords]" or "click[value]").
3.  **State & Preconditions**: Explicitly mention necessary preconditions (e.g., "Verify the price is below the target before purchasing").
4.  **Error Handling**: Include a rule for what to do if the agent encounters an error or unexpected situation.
5.  **Product Click Format (MANDATORY)**: When instructing the agent to open a product from the search results page, ALWAYS use the product's ASIN code (e.g., `click[B07WSW2FTQ]`). The ASIN is the short alphanumeric code (starts with B0) shown before each product title in the search results. Product titles, names, and descriptions are NOT clickable — `click[<item title>]` or `click[<product name>]` is an INVALID action and the page will not change.

# Input Data
1.  **Task Description:**
{task}
2.  **Relevant Skill Contents:**
{skill_contents}

# Examples
Here are some examples of well-structured Procedural Guidance texts you should emulate (focus on style, structure, and level of detail instead of content):
{overall_procedure_examples}

# IMPORTANT CORRECTION TO THE EXAMPLES ABOVE
The examples above come from a related environment and some of them describe clicking a product by its full title or name (for example `click[{{item_title}}]`, `click[<item_name>]`, or `click[<Full Name of Item>]`).
That format is INVALID in THIS WebShop environment: clicking a product title produces an invalid action and the page does not change, wasting steps until timeout.
In THIS environment, every product on the search results page is preceded by its ASIN code — a short alphanumeric identifier (starts with B0) placed before the product title (e.g., B08BTJ1N6V). The ASIN code is the ONLY valid click target for opening a product page.
Therefore, in the Procedural Guidance you generate:
- ALWAYS instruct the agent to use `click[<ASIN>]` (e.g., `click[B08BTJ1N6V]`) to open a product from search results.
- NEVER instruct the agent to click a product title, name, or description.
- Include an error-handling rule: if an action had no effect (the observation did not change), the clicked target was invalid; the agent must re-read the observation and click the ASIN code instead.

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure> your generated overall procedure guidance here </Overall_Procedure>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]

def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
4.  **Error Handling**: Implement robust error handling for LLM calls and environment interactions.
5.  **Function Naming**: Name the function `overall_procedure_code` as per the template.
6.  **Invalid Action Detection & Correction**: In the template's "RUNTIME ANALYSIS & INTERVENTION" section, implement stuck detection: keep track of the previous observation (a variable initialized before the loop), and when the current observation is exactly identical to the previous one, the last action was INVALID and had no effect — in that case append a user-message hint telling the agent: the previous action was invalid; on search result pages each product is preceded by its ASIN code (a short code starting with B0, e.g., B08BTJ1N6V), and the agent must click that ASIN code (e.g., `click[B08BTJ1N6V]`), NOT the product title or description. Do NOT use `continue`, `break`, or `return` in that section.

# Input Data
1.  **Task Description:**
{task}
2.  **Overall Procedural Guidance:**
{overall_procedure}
3.  **Procedure Code Template:**
```python
{procedure_code_template}
```

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure_Code> your generated overall procedure workflow code here </Overall_Procedure_Code>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]
