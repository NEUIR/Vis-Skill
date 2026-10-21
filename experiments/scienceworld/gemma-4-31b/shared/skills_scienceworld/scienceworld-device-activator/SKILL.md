---
name: scienceworld-device-activator
description: Activates a device (e.g., blast furnace, stove) to initiate a process like heating. Use this skill when you have placed materials inside a device and need to start its operation. Takes a device name as input and outputs a confirmation of activation, enabling tasks that require energy input or material processing to progress.
---
# Skill: Device Activator

## Purpose
Activate a device (turn it on, start it, fire it up) to begin its operation — typically heating, processing, or powering a task step.

## Core Workflow
1. **Verify Contents:** `look at <DEVICE_NAME>` to confirm materials are placed inside.
2. **Check State:** Observation should show "which is turned off" or "which is deactivated."
3. **Activate:** `activate <DEVICE_NAME>`
4. **Verify Activation:** `look at <DEVICE_NAME>` — confirm it now reads "turned on" or "activated."
5. **Monitor (if needed):** Use measurement tools (e.g., `use thermometer on <MATERIAL>`) to track progress.
6. **Deactivate When Done:** `deactivate <DEVICE_NAME>`

## Key Actions
| Action | Purpose |
|--------|---------|
| `look at <DEVICE>` | Verify contents and device state |
| `activate <DEVICE>` | Turn on / start the device |
| `deactivate <DEVICE>` | Turn off the device when done |
| `use TOOL on OBJ` | Monitor material state during processing |

## Example
**Task:** Melt lead in a blast furnace.

1. `look at blast furnace` — confirms: metal pot with lead inside, furnace is turned off
2. `activate blast furnace`
3. `look at blast furnace` — confirms: "which is turned on"
4. `use thermometer on lead` — monitor temperature
5. `look at blast furnace` — observe: "a substance called liquid lead" (melting complete)
6. `deactivate blast furnace`

## Important Notes
* Most devices are pre-opened. Do not use `open` or `close` unless the observation indicates a closed door.
* Always verify activation succeeded before proceeding to the next task step.
