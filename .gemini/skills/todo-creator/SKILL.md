---
name: todo-creator
description: Decomposes a project plan (e.g., plan.md) into actionable TODO files in ./gemini_artifacts/todo. Use during the planning phase to translate high-level strategies into a sequence of discrete, implementable tasks.
---

# Todo Creator

## Overview

The `todo-creator` skill bridges the gap between high-level planning and autonomous implementation. It transforms a broad roadmap into a structured queue of tasks that the `implementor` skill can subsequently execute.

## Workflow

When a plan is finalized or provided, follow these steps:

### 1. Plan Analysis
- **Read Plan**: Analyze the `plan.md` or the user's provided implementation strategy.
- **Identify Tasks**: Extract distinct, logical units of work. Each task should be significant enough to require its own implementation and testing cycle, but small enough to be manageable.

### 2. TODO Generation
- **Target Directory**: All files MUST be created in `./gemini_artifacts/todo/`.
- **Naming Convention**: Use a numerical prefix to establish implementation order (e.g., `001-setup-data-loader.md`, `002-implement-fft.md`).
- **File Structure**: Each TODO file must contain:
    - **Task Name**: A concise title for the task.
    - **Description**: A detailed explanation of what needs to be implemented.
    - **System Impact**: A section describing which files, modules, or behaviors will be affected by this change.
    - **Subtasks (Optional)**: A checklist of initial steps (the `implementor` will further refine these).

## Example TODO Format

```markdown
# [00X] Task Name

## Description
Detailed explanation of the feature or fix to be implemented.

## System Impact
- `core.py`: Modified to include X logic.
- `loader.py`: New function Y added.
- `tests/`: New test file `test_X.py` created.

## Subtasks
- [ ] Initial step 1
- [ ] Initial step 2
```
