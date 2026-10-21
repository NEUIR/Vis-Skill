---
name: scienceworld-circuit-builder
description: This skill constructs a simple electrical circuit by connecting components like batteries, wires, and light bulbs. Use when the agent needs to test electrical conductivity or create a functional circuit for an experiment. The input is a set of available components, and the output is a fully connected circuit ready for activation.
---
# Instructions

## 1. Objective
Your goal is to build a functional electrical circuit to test the conductivity of a target object (e.g., a metal pot). The circuit's success is indicated by a light bulb turning on.

## 2. Prerequisites & Setup
*   **Location:** You must be in the **workshop**. If not, teleport there.
*   **Components:** Ensure the following are present on the table:
    *   A battery
    *   A light bulb (e.g., blue light bulb)
    *   At least two wires (e.g., orange, yellow, green)
    *   The target object to test (e.g., metal pot)
    *   A blue box (for conductive objects) and an orange box (for non-conductive objects)
*   **Inventory:** The target object must be in your inventory. If not, locate and pick it up.

## 3. Core Procedure
Follow these steps to construct the series circuit. Use the exact `connect` actions as shown.

1.  **Place the Target Object:** Drop the target object from your inventory onto the floor of the workshop.
2.  **Connect Power Source:**
    *   `connect battery anode to <Wire1> terminal 1` (e.g., `orange wire`)
    *   `connect battery cathode to <Wire2> terminal 1` (e.g., `yellow wire`)
3.  **Connect Light Bulb:**
    *   `connect <Wire1> terminal 2 to <LightBulb> cathode` (e.g., `blue light bulb`)
    *   `connect <Wire3> terminal 2 to anode in <LightBulb>` (e.g., `green wire`)
4.  **Integrate Target Object into Circuit:**
    *   `connect <TargetObject> terminal 1 to <Wire2> terminal 2`
    *   `connect <TargetObject> terminal 2 to <Wire3> terminal 1`

## 4. Testing & Conclusion
1.  **Observe:** After completing the connections, `wait1` for 1-2 iterations.
2.  **Check Result:** `look around`. If the light bulb is **on**, the target object is **electrically conductive**.
3.  **Final Action:** Based on the result:
    *   If **conductive**: `move <TargetObject> to blue box`
    *   If **non-conductive**: `move <TargetObject> to orange box`

## 5. Key Principles
*   **Circuit Logic:** This builds a simple series circuit: Battery -> Wire1 -> Light Bulb -> Wire3 -> Target Object -> Wire2 -> Battery.
*   **Action Precision:** Use the exact object names and connection points (anode, cathode, terminal 1/2) as observed in the environment.
*   **Error Handling:** If a component is missing, examine the room (`look around`) to identify available substitutes before proceeding.

## 6. Example
**Task:** Test whether a metal pot is electrically conductive.
1. `teleport to workshop`
2. `pick up metal pot`
3. `connect battery anode to orange wire terminal 1`
4. `connect battery cathode to yellow wire terminal 1`
5. `connect orange wire terminal 2 to blue light bulb cathode`
6. `connect green wire terminal 2 to anode in blue light bulb`
7. `connect metal pot terminal 1 to yellow wire terminal 2`
8. `connect metal pot terminal 2 to green wire terminal 1`
9. `wait1` — observe if the light bulb turns on
10. Light bulb is on → `move metal pot to blue box` (conductive)
