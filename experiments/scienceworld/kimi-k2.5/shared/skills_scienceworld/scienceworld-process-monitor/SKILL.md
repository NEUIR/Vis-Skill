---
name: scienceworld-process-monitor
description: This skill observes the state of an active apparatus and its contents to track progress. Use when you need to periodically check for state changes (e.g., solid to liquid) during a heating or reaction process. The skill uses 'look at' or 'examine' actions on the apparatus and substance.
---
# Process Monitoring Protocol

## When to Use
Activate when an apparatus (stove, burner) is active and you need to check whether a substance has undergone a state change (e.g., solid to liquid, liquid to gas).

## Procedure
1. `look at <APPARATUS>` — verify it is active and contains the target substance.
   - If apparatus is not active, use `activate <APPARATUS>` first.
   - If substance is missing, check inventory or nearby containers.
2. `examine <SUBSTANCE>` — check its current state description.
3. Compare the observed state to the previous state. If changed (e.g., "chocolate" to "liquid chocolate"), proceed to the next task step.
4. If no change, use `wait` then repeat from step 1.

Run every few steps during an active process. This skill only observes — it does not modify the apparatus or handle substances.

## Example
**Task:** Monitor chocolate melting on a stove.
1. `look at stove` — observation: "a stove (activated), containing: chocolate"
2. `examine chocolate` — observation: "chocolate"
3. No change yet. `wait`
4. `examine chocolate` — observation: "liquid chocolate"
5. State change detected. Proceed to take a temperature measurement.
