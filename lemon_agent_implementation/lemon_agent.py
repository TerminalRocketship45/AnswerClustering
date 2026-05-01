"""
Lemon Agent — direct Python translation of the Lisp find-failure-modes function.

Original Lisp (Mark Klein, MIT):

    (defun find-failure-modes (solution &key hint)
      (let* ((criteria (find-criteria solution))
             (purpose  (find-if #'(lambda (p) (isa-p p '(goal barrier cause)))
                                (gbsmcontext solution)))
             (prompt   (with-output-to-string (stream) ...))
             (response (askLLM prompt))
             (sexp     (parse-llm-output response)))
        sexp))

This module is a structural translation of that Lisp function into Python.
It preserves the same logical steps: criteria lookup, purpose extraction,
prompt construction, LLM invocation, and output parsing.
"""

from __future__ import annotations
from typing import Optional, Any

from models import Solution, Criterion, Purpose
from gbsm import find_criteria, find_purpose, explain_purpose
from llm_client import ask_llm, parse_llm_output


def _build_prompt(
    purpose: Optional[Purpose],
    criteria: list[Criterion],
    solution: Solution,
    hint: Optional[str],
) -> str:
    """Reconstruct the Lisp prompt-building block in Python with comments."""

    # Start with an empty list of prompt lines, like accumulating text in a string.
    lines: list[str] = []

    # Add a description of the purpose, as the Lisp code does with explain-purpose.
    lines.append(explain_purpose(purpose))
    lines.append("")  # blank line for readability in the prompt

    # Begin the instruction that asks for failure modes.
    lines.append("I want you to identify the different ways that we can fail to ")

    # Add the actual target of failure analysis depending on the purpose type.
    if purpose is None:
        # If no purpose was found, fall back to a generic objective.
        lines.append("achieve the intended objective.")
    elif purpose.ptype == "goal":
        # If the purpose is a goal, ask about failing to achieve that goal.
        lines.append(f"achieve the goal {purpose.name}")
    else:
        # If the purpose is a barrier or cause, ask about overcoming the problem.
        lines.append(f"Overcome the problem {purpose.name}")

    lines.append("")

    # Explain that the following lines list the criteria for a good solution.
    lines.append(
        f"These are the criteria that a good solution for this {purpose.ptype if purpose else 'objective'} "
        "would satisfy:"
    )

    # Add each criterion line to the prompt, similar to the Lisp loop.
    for c in criteria:
        description_text = f" {c.description}" if c.description else ""
        lines.append(f"- {c.eid} {c.name}{description_text}")

    lines.append("")

    # Ask the model to enumerate failure modes for each criterion.
    lines.append(
        "I want you to enumerate, for each of the criteria, the ways that we can fail to "
        "achieve these criteria."
    )
    lines.append("")

    # Ask the model to rate how likely each failure mode is for the proposed solution.
    lines.append(
        "And I want you to tell me, for each failure mode, whether that failure is "
        "highly likely to occur if we use the following solution:"
    )

    # Include the solution name and optional description.
    lines.append(f"{solution.name}{' ' + solution.description if solution.description else ''}")

    if hint:
        # If a hint is provided, include it in the prompt.
        lines.append("")
        lines.append(
            f"Do your best to incorporate this hint when defining the failure modes: {hint}"
        )

    lines.append("")

    # Instruct the LLM to format the output as JSON following the schema.
    lines.append(
        "Give me your response as a JSON structure that gives the list of failure modes you found, as follows:"
    )
    lines.append("```json")
    lines.append("[")
    lines.append('{')
    lines.append('  "type": "failure",')
    lines.append('  "criterionName": <the name for the criterion affected by the failure mode>,')
    lines.append('  "criterionID": <the ID for the criterion affected by the failure mode>,')
    lines.append('  "description": <a 2 or 3 sentence description of this proposed failure mode>,')
    lines.append('  "name": <a 4 to 6 word name for the failure mode>,')
    lines.append('  "risk": <say high if the solution is at high risk for this failure, and low if the solution is at low risk>,')
    lines.append('  "rationale": <a 2 or 3 sentence description of WHY you gave that risk rating for that failure mode>')
    lines.append('}]')
    lines.append("```")

    # Join all prompt lines into a single string separated by newlines.
    return "\n".join(lines)


def find_failure_modes(
    solution: Solution,
    hint: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Direct translation of the Lisp defun find-failure-modes."""

    # Step 1: collect the criteria for the given solution.
    criteria: list[Criterion] = find_criteria(solution)

    # Step 2: find the purpose object from the solution context.
    purpose: Optional[Purpose] = find_purpose(solution)

    # Step 3: build the prompt text that will be sent to the LLM.
    prompt: str = _build_prompt(purpose, criteria, solution, hint)

    # Step 4: send the prompt to the LLM and get the raw response.
    response: str = ask_llm(prompt)

    # Step 5: parse the LLM output into structured failure mode objects.
    failure_modes: list[dict[str, Any]] = parse_llm_output(response)

    # Return the parsed failure modes to the caller.
    return failure_modes
