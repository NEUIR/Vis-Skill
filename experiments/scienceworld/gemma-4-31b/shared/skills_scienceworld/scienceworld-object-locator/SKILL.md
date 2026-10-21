---
name: scienceworld-object-locator
description: Searches for a specific target object across multiple rooms by systematically teleporting to likely locations and examining each room. Use this skill when you need to find an object whose exact location is unknown. Iterates through candidate rooms using teleport and look around, checks for the object, and continues until found, returning the object's location.
---
# Instructions

## Purpose
Locate a specific target object within the ScienceWorld environment when its location is unknown.

## Inputs
- **Target Object:** The name of the object to find (e.g., "thermometer", "metal fork").
- **Candidate Rooms (Optional):** A list of room names to search. If not provided, the skill will use a default priority list based on common object locations.

## Core Process
1.  **Initialize:** Receive the target object name.
2.  **Plan Search Order:** Determine the sequence of rooms to search. Use the provided `prioritize_rooms.py` script to generate an optimal search order based on the target object.
3.  **Execute Search Loop:**
    a. Teleport to the next room in the search order.
    b. Use the `look around` action to observe the room's contents.
    c. Parse the observation to check for the presence of the target object.
    d. If the object is found, **stop the search** and output its location.
    e. If the object is not found, proceed to the next room.
4.  **Output:** Report the room where the object was found. If the object is not found after searching all candidate rooms, report this failure.

## Key Actions
- `teleport to <ROOM_NAME>`
- `look around`
- `examine <OBJECT>` (if more detail is needed to confirm identity)
- `pick up <OBJECT>` (if the goal is to acquire the object, not just locate it)

## Example
**Task:** Find the thermometer in the environment.

1. `teleport to kitchen`
2. `look around` — no thermometer found
3. `teleport to workshop`
4. `look around` — observation includes "a thermometer, currently reading a temperature of 10 degrees celsius"
5. Thermometer located in workshop. `pick up thermometer` if needed.

## Notes & Best Practices
- **Efficiency:** Always `look around` immediately after teleporting to get the full room state.
- **Parsing:** Observations list objects after "you see:". Check this list for the target object's name.
- **Confirmation:** If an object's name is ambiguous (e.g., "fork" could be "metal fork" or "plastic fork"), use `examine` to get a detailed description.
- **Inventory:** If the task requires using the object, remember to `pick it up` after finding it.
- **Error Handling:** If an object is not in its most likely room, the systematic search will eventually find it. Do not give up after the first few rooms.
