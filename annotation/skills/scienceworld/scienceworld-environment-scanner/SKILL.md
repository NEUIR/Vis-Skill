---
name: scienceworld-environment-scanner
description: This skill performs an initial observation of the current location to identify available objects, containers, and connections to other rooms. Use this when starting a new task in ScienceWorld to gather situational awareness, especially when the agent needs to understand its surroundings before taking action. It outputs a detailed room description, listing visible items and accessible doors.
---
# Instructions

Execute this skill when you first arrive at a location in ScienceWorld or when you need to refresh your understanding of the current environment.

## Core Action
1.  **Always begin by using the `look around` action.** This is the primary and mandatory step of this skill. It provides the foundational observation of the room.

## Output Interpretation
After executing `look around`, analyze the observation. Your analysis must structure the output clearly:
*   **Current Room:** State the name of the room you are in.
*   **Visible Objects & Substances:** List all items and substances reported in the observation.
*   **Accessible Doors:** List all doors mentioned, noting their destination and whether they are open or closed.
*   **Initial Assessment:** Briefly note any items of immediate interest or relevance to your known task.

## Example Output Format
