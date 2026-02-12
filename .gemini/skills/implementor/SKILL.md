---
name: implementor
description: Autonomously implements features from TODO files in ./gemini_artifacts/todo. Use when there are pending implementation tasks that require a structured process of planning, coding, multi-path testing (happy, hard, and fail paths), and documentation updates.
---

# Implementor

## Overview

The `implementor` skill follows a rigorous, document-driven workflow to transform planned tasks into verified, documented code. It ensures that every change is grounded in design, verified through comprehensive testing, and reflected in the project's documentation.

## Workflow

When triggered, follow these steps for each TODO file in `./gemini_artifacts/todo`, following their numerical or chronological order:

### 1. Planning & Subtasking
- **Read Context**: Read `design.md` and `plan.md` to understand the architectural requirements and the high-level roadmap.
- **Decompose**: Breakdown the main task in the TODO file into granular, actionable subtasks.
- **Annotate**: Add these subtasks to the TODO file if they aren't already present.

### 2. Test-Driven Implementation
- **Implement**: Write the code to fulfill the subtasks. Aim for modular, testable units.
- **Verification**:
    - **Happy Path**: Test the standard, expected usage.
    - **Hard Path**: Test edge cases, boundary conditions, and complex state transitions.
    - **Should-Fail Path**: Test error handling by providing invalid input or forcing failure conditions.
- **Fix**: Run tests immediately. If they fail, debug and iterate until all paths pass.

### 3. Completion & Documentation
- **Review**: Ensure all subtasks in the TODO file are marked as complete (crossed-out or checked). If anything remains, implement it.
- **Update Design**: Reflect any architectural or interface changes in `design.md`. Ensure the documentation matches the new reality of the codebase.

## Testing Strategy

| Path | Goal | Example |
| :--- | :--- | :--- |
| **Happy** | Verify core functionality works as intended. | Correct input returns correct output. |
| **Hard** | Verify robustness against unusual but valid data. | Empty lists, maximum values, concurrent access. |
| **Should-Fail** | Verify graceful degradation and error reporting. | Invalid types, missing files, unauthorized access. |
