# RooCode Orchestrator Mode Prompt Engineering Guide

## Introduction

This document is designed to help large language models (LLMs) write clear, actionable, and interpretable instructions for RooCode Orchestrator Mode. By following these guidelines, LLMs can maximize the effectiveness of the Orchestrator, ensuring that complex tasks are broken down, delegated, and completed with precision and clarity.

## Overview of RooCode Orchestrator Mode

RooCode Orchestrator Mode acts as a high-level coordinator within the RooCode system. Its primary responsibilities include:

- Decomposing complex objectives into logical, manageable subtasks.
- Delegating each subtask to the most appropriate specialized mode (e.g., Code, Debug, Architect).
- Tracking the progress and outcomes of all subtasks.
- Synthesizing results into a coherent final output.

**Clarity, explicitness, and structured instructions are essential** for the Orchestrator to function optimally. Ambiguous or incomplete instructions can lead to misinterpretation, inefficiency, or failure to achieve the desired outcome.

## Principles of Effective Prompt Engineering for Orchestrator Mode

- **Use unambiguous, direct language:** Avoid vague terms and ensure every instruction is clear.
- **Specify goals, constraints, and success criteria:** Each subtask should have a well-defined objective, boundaries, and explicit criteria for completion.
- **Avoid assumptions:** State all necessary context, requirements, and dependencies explicitly.
- **Prefer step-by-step instructions:** For complex workflows, break down the process into sequential, logical steps.

## Structuring Instructions for Maximum Interpretability

- **Use clear headings, bullet points, and numbered lists:** Structure enhances readability and interpretability.
- **For each subtask, provide:**
  - **Specific goal:** What is to be accomplished.
  - **Relevant context:** Include all necessary information from the parent task or previous subtasks.
  - **Scope boundaries:** Define what is in scope and what is explicitly out of scope.
  - **Explicit completion criteria:** State how the Orchestrator should determine the subtask is done (e.g., use of `attempt_completion`).
- **Signal completion and summarize outcomes:** Instruct the Orchestrator to clearly indicate when a subtask or the overall task is complete, and to provide a concise summary of results.

## Breaking Down Complex Tasks

- **Decompose large objectives:** Break down the main goal into logical, atomic subtasks that can be independently delegated.
- **Delegate to specialized modes:** Assign each subtask to the mode best suited for its requirements (e.g., debugging to Debug mode, code changes to Code mode).
- **Good breakdown example:**
  1. Analyze the requirements and identify all necessary components.
  2. Implement each component in the appropriate mode.
  3. Integrate components and perform end-to-end testing.
  4. Summarize and signal completion.
- **Bad breakdown example:** "Fix all issues and make it work." (Too vague, lacks structure and delegation.)

## Specifying Subtask Boundaries and Context

- **Define information passed to each subtask:** Clearly state what data, files, or context should be available.
- **Avoid ambiguity:** Be explicit about what is and is not included in the subtask’s scope.
- **Supersede general mode instructions when needed:** If your instructions differ from the default behavior of a mode, state this explicitly.

## Providing Explicit Completion Criteria

- **Always specify completion signals:** Instruct the Orchestrator to use `attempt_completion` or another explicit mechanism to indicate when a subtask is finished.
- **Require concise, thorough summaries:** Each completion signal should include a summary of what was accomplished, any issues encountered, and the final outcome.

## Common Pitfalls and How to Avoid Them

- **Vague instructions:** Always be specific and direct.
- **Overlapping or unclear subtask boundaries:** Ensure each subtask is atomic and non-overlapping.
- **Missing context or implicit assumptions:** Provide all necessary information explicitly.
- **Failing to specify completion signals:** Always instruct the Orchestrator on how to signal completion.

## Example Prompts and Anti-Patterns

### Well-Structured Example

```
# Task: Implement and document a new API endpoint

1. **Design the API schema**
   - Goal: Define the request/response structure for the new endpoint.
   - Context: The endpoint will allow users to submit feedback.
   - Scope: Only design the schema, do not implement logic.
   - Completion: Use `attempt_completion` with a summary of the schema.

2. **Implement the endpoint logic**
   - Goal: Write the backend code for the feedback submission endpoint.
   - Context: Use the schema from the previous step.
   - Scope: Do not write documentation or tests yet.
   - Completion: Use `attempt_completion` with a summary of the implementation.

3. **Write API documentation**
   - Goal: Document the new endpoint for external developers.
   - Context: Reference the schema and implementation details.
   - Scope: Only documentation, no code changes.
   - Completion: Use `attempt_completion` with a summary of the documentation.

4. **Signal overall completion**
   - Summarize all outcomes and signal final completion.
```

**Why this works:** Each subtask is atomic, has clear goals, context, boundaries, and explicit completion criteria.

### Poorly Structured Example

```
Implement a feedback API and document it. Make sure it works.
```

**Why this fails:** The instruction is vague, lacks structure, does not specify subtasks, context, or completion criteria.

## Final Checklist for LLMs Writing Instructions

- [ ] Are all goals, constraints, and success criteria explicitly stated?
- [ ] Is each subtask atomic, non-overlapping, and delegated to the correct mode?
- [ ] Is all necessary context provided for each subtask?
- [ ] Are scope boundaries (in/out of scope) clearly defined?
- [ ] Are explicit completion signals (e.g., `attempt_completion`) required for each subtask?
- [ ] Are instructions structured with clear headings, lists, and summaries?
- [ ] Have you avoided vague language and implicit assumptions?
- [ ] Is there a final summary and completion signal for the overall task?

## Conclusion

Clarity, explicitness, and structured instructions are essential for maximizing the effectiveness of RooCode Orchestrator Mode. By following the guidance in this document, LLMs can ensure that their instructions are interpretable, actionable, and lead to successful task completion.