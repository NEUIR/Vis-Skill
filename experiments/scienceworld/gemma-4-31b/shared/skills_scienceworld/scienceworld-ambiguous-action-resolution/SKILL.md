---
name: scienceworld-ambiguous-action-resolution
description: Use when the ScienceWorld environment returns an "Ambiguous request" prompt with a numbered list of identical action options. This skill resolves the disambiguation by selecting the lowest available number (typically 0) to proceed, ensuring task progression when multiple identical object instances exist and the system cannot automatically determine which instance to act upon.
---
# Skill: scienceworld-ambiguous-action-resolution

## Purpose

Resolve system disambiguation prompts that block task progression when the ScienceWorld environment cannot determine which identical object instance the agent intends to act upon. This is a mechanical bypass, not a decision-making step.

## When to Use

- When the environment returns an observation containing `"Ambiguous request: Please enter the number for the action you intended (or blank to cancel):"` followed by a numbered list
- When multiple identical object instances exist (e.g., five identical banana seeds in a jar) and the system requires explicit selection
- Only when options are functionally identical -- do NOT use this skill for menu selections where options have different meanings

## Workflow

1. **Recognize the prompt** -- Identify the exact phrase: `"Ambiguous request: Please enter the number for the action you intended (or blank to cancel):"`.
2. **Scan the options** -- Confirm that the listed options (0, 1, 2...) are functionally identical, referring to different instances of the same object.
3. **Select the lowest number** -- Choose `0` (the first option) to proceed. Any valid selection will produce the same result.
4. **Execute** -- Output the selected number as the next action.

## Examples

### Example 1: Moving one of several identical banana seeds

**Observation:**
```
Ambiguous request: Please enter the number for the action you intended (or blank to cancel):
0: move banana seed (in seed jar, in inventory, in agent, in greenhouse) to flower pot 1 (in greenhouse)
1: move banana seed (in seed jar, in inventory, in agent, in greenhouse) to flower pot 1 (in greenhouse)
2: move banana seed (in seed jar, in inventory, in agent, in greenhouse) to flower pot 1 (in greenhouse)
```

**Action:**
```
> 0
```

The system proceeds with moving one banana seed to flower pot 1.

### Example 2: Picking up one of several identical items

**Observation:**
```
Ambiguous request: Please enter the number for the action you intended (or blank to cancel):
0: pick up red wire (on table, in workshop)
1: pick up red wire (on table, in workshop)
```

**Action:**
```
> 0
```

## Key Principles

- **Efficiency** -- Do not overthink; the objects are identical and any choice is valid.
- **Consistency** -- Always default to `0` unless task logic specifically requires a different instance.
- **Scope** -- This skill is strictly a disambiguation mechanic, not a decision-making process.

## Anti-Patterns

- Do NOT use for non-ambiguous choices where options have different meanings.
- Do NOT trigger if the observation does not contain the exact "Ambiguous request" phrase.
- Do NOT waste steps analyzing differences between identical instances.
