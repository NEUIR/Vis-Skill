---
name: alfworld-object-placer
description: Places a held object onto or into a target receptacle. Trigger this skill when the agent is carrying an object and needs to deposit it at a specific location to complete a task, such as installing an item on a holder or storing it in a container. It takes the object and destination receptacle as inputs, performing a 'put' action to finalize the object's positioning according to the task goal.
---
# Instructions
1. **Verify Prerequisites**:
   - Confirm you are currently holding the target object (`{obj}`).
   - Confirm the target receptacle (`{recep}`) is present and accessible in the environment.
   - If the receptacle is a container (e.g., cabinet, drawer), ensure it is open.

2. **Navigate to Receptacle**:
   - Use the `go to {recep}` action to move to the target receptacle's location.

3. **Execute Placement**:
   - Use the `put {obj} in/on {recep}` action to deposit the held object.
   - The phrasing (`in` vs `on`) is determined by the receptacle type (e.g., `in` for containers, `on` for surfaces/holders). Use the trajectory as a reference.

4. **Handle Errors**:
   - If the environment responds with "Nothing happened", the action was invalid. Consult the `references/action_guide.md` for troubleshooting common failure modes.
