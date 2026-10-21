---
name: alfworld-object-retriever
description: This skill picks up a target object from a specified receptacle after the object has been visually confirmed. Use this skill when the agent has located an object in a receptacle and needs to acquire it. It requires the object and source receptacle as inputs and results in the object being added to the agent's inventory.
---
# Instructions

Use this skill to acquire a target object from a known source receptacle in the ALFWorld environment.

## Prerequisites
1. **Visual Confirmation**: You must have directly observed the target object within the specified source receptacle in a prior observation (e.g., "On the {recep}, you see a {obj}...").
2. **Proximity**: You should be at or near the location of the source receptacle.

## Core Action Sequence
Execute the following action precisely:
