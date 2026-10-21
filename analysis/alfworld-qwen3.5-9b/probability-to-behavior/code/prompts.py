"""
Prompt fragments shared by state extraction, probing and rendering.

The system prompt is byte-identical to the one used by the two main
experiments (`text-skill/code/run_text.py`, `vis-skill/code/run_visual.py`), so
the decision states we probe sit in exactly the same conditioning context the
main runs used.

The skill-injection wrapper is held as close to identical between the text and
the visual condition as the modality allows — the only thing that differs is
whether the skill body arrives as characters or as pixels. That is the whole
point of the comparison: "does the same keyword-frequency interference act
differently in text and in images?".

The one unavoidable wording difference is the intro line: the visual condition
has to tell the model that the documents are the images below (SKILL_INTRO_TEXT
vs SKILL_INTRO_TEXT_VISUAL). The closing instruction (SKILL_OUTRO_TEXT) and
everything above the skill turn are byte-identical across conditions.
"""

SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""


# Text that opens the skill-reference turn (identical in both conditions).
SKILL_INTRO_TEXT = "\nTASK SKILL REFERENCE:"

SKILL_INTRO_TEXT_VISUAL = "\nTASK SKILL REFERENCE (the skill documents are the images below):"

# Text that closes the skill-reference turn (identical in both conditions).
SKILL_OUTRO_TEXT = (
    "\n\nBased on the above skill reference and the current situation, decide your "
    "next action. Reply with exactly one action in the required format."
)

# Prefix used when force-decoding a candidate action, so that every candidate is
# scored under a byte-identical prefix and the resulting log-probabilities are
# directly comparable / normalisable into a distribution.
FORCED_THOUGHT = "Thought: I will now perform the action."
