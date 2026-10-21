---
name: scienceworld-container-relocator
description: Moves an object from inventory to a specified container in a target room. Use when the task requires placing an item into a particular receptacle (e.g., 'move it to the orange box').
---
# Skill: Container Relocator

## Procedure
1. Confirm the object is in your inventory. If not, acquire it first.
2. `teleport to <ROOM>` — navigate to the room containing the target container.
3. `move <OBJECT> to <CONTAINER>` — transfer the object to the destination.
4. `look at <CONTAINER>` — verify the object is now inside.

## Example Flow (from trajectory)
1.  Task: "move it to the orange box in the workshop."
2.  Identify: Object = dove egg (in inventory), Container = orange box, Room = workshop.
3.  Action: `teleport to workshop`
4.  Action: `move dove egg to orange box`
