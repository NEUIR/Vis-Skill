---
name: scienceworld-circuit-connector
description: This skill connects two electrical components (e.g., wires, batteries, devices) by their terminals to build or modify a circuit. Use when constructing electrical setups for testing, such as conductivity checks or device activation. The input includes two component identifiers and their connection points, and the output is an established electrical connection.
---
# Instructions

## Primary Action
Execute the `connect OBJ to OBJ` action. The syntax is `connect <ComponentA> <TerminalA> to <ComponentB> <TerminalB>`.

## When to Use
Use this skill when you need to establish an electrical connection between two components as part of building a circuit for testing or device operation. Common scenarios include:
*   Building a circuit to test a material's conductivity.
*   Connecting a power source (battery) to a load (light bulb, buzzer).
*   Integrating a switch or sensor into a circuit.

## Procedure
1.  **Identify Components:** Locate the two components you need to connect (e.g., `battery`, `black wire`, `unknown substance M`, `electric buzzer`).
2.  **Identify Terminals:** Examine the components if necessary to determine their available connection points (e.g., `anode`, `cathode`, `terminal 1`, `terminal 2`).
3.  **Formulate Command:** Construct the `connect` command using the pattern: `connect <ComponentA> <TerminalA> to <ComponentB> <TerminalB>`.
    *   Example from trajectory: `connect battery anode to black wire terminal 1`
4.  **Execute:** Issue the command. A successful observation will confirm the connection (e.g., "`anode on battery is now connected to terminal 1 on black wire`").

## Key Principles
*   **Circuit Logic:** Ensure your connections follow basic electrical principles. A typical series circuit path might be: Battery (+) -> Wire -> Device -> Wire -> Battery (-).
*   **Verification:** After making connections, check the state of indicators (like lights or buzzers) or use the `examine` action on components to verify the circuit is complete and functional.
*   **Error Handling:** If a connection fails, verify the component and terminal names are correct. Some terminals may only accept one connection.

## Related Skills
*   For disconnecting components, use the `disconnect OBJ` action.
*   To activate/deactivate a powered component in the circuit, use the `activate OBJ`/`deactivate OBJ` actions.
*   Refer to `references/circuit_basics.md` for common circuit diagrams and troubleshooting.
