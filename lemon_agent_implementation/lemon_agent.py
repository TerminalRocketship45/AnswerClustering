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

The function:
  1. Fetches the evaluation criteria for the solution (cost, risk, feasibility …).
  2. Finds the purpose (goal / barrier / cause) that motivates the solution.
  3. Builds a detailed prompt that asks the LLM to enumerate failure modes —
     one per criterion — and rate each as HIGH or LOW risk for this solution.
  4. Calls the LLM and parses the JSON response.
  5. Returns the list of FailureMode dicts.

"Lemons" in the original system are failure modes rated HIGH risk.  The caller
can count lemons to decide whether to eliminate a solution from the candidate set.
"""

from __future__ import annotations
from typing import Optional, Any

from models import Solution, Criterion, Purpose
from gbsm import find_criteria, find_purpose, explain_purpose
from llm_client import ask_llm, parse_llm_output


# ---------------------------------------------------------------------------
# Prompt builder  (mirrors the with-output-to-string block in the Lisp)
# ---------------------------------------------------------------------------

def _build_prompt(
    purpose: Optional[Purpose],
    criteria: list[Criterion],
    solution: Solution,
    hint: Optional[str],
) -> str:
    """
    Reconstruct the Lisp prompt-building logic in Python.

    The Lisp code uses (pr ...) to stream formatted text into a string.
    Here we build the same string with an f-string / join approach.
    """
    lines: list[str] = []

    # --- Purpose context (explain-purpose) ---
    lines.append(explain_purpose(purpose))
    lines.append("")

    # --- Ask to enumerate failure modes ---
    lines.append("I want you to identify the different ways that we can fail to ")

    if purpose is None:
        lines.append("achieve the intended objective.")
    elif purpose.ptype == "goal":
        lines.append(f"achieve the goal: {purpose.name}")
    else:  # barrier or cause
        lines.append(f"overcome the problem: {purpose.name}")

    lines.append("")

    # --- Criteria list ---
    ptype_label = purpose.ptype if purpose else "objective"
    lines.append(
        f"These are the criteria that a good solution for this {ptype_label} "
        "would satisfy:"
    )
    for c in criteria:
        desc = f" {c.description}" if c.description else ""
        lines.append(f"- {c.eid} {c.name}{desc}")

    lines.append("")

    # --- Instruction to enumerate per-criterion failure modes ---
    lines.append(
        "I want you to enumerate, for each of the criteria, "
        "the ways that we can fail to achieve these criteria."
    )
    lines.append("")
    lines.append(
        "And I want you to tell me, for each failure mode, whether that failure "
        "is highly likely to occur if we use the following solution:"
    )
    sol_desc = f" {solution.description}" if solution.description else ""
    lines.append(f"{solution.name}{sol_desc}")

    # --- Optional hint ---
    if hint:
        lines.append("")
        lines.append(
            f"Do your best to incorporate this hint when defining the "
            f"failure modes: {hint}"
        )

    # --- JSON schema instruction ---
    lines.append("")
    lines.append(
        "Give me your response as a JSON structure that gives the list of "
        "failure modes you found, as follows:"
    )
    lines.append("```json")
    lines.append("[")
    lines.append('{ ')
    lines.append('  "type": "failure",')
    lines.append('  "criterionName": <the name for the criterion affected by the failure mode>,')
    lines.append('  "criterionID": <the ID for the criterion affected by the failure mode>,')
    lines.append('  "description": <a 2 or 3 sentence description of this proposed failure mode>,')
    lines.append('  "name": <a 4 to 6 word name for the failure mode>,')
    lines.append('  "risk": <say high if the solution is at high risk for this failure, and low if the solution is at low risk>,')
    lines.append('  "rationale": <a 2 or 3 sentence description of WHY you gave that risk rating for that failure mode>')
    lines.append('}]')
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# find_failure_modes  (direct translation of the Lisp defun)
# ---------------------------------------------------------------------------

def find_failure_modes(
    solution: Solution,
    hint: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Python translation of:

        (defun find-failure-modes (solution &key hint) ...)

    Returns a list of failure-mode dicts, each with keys:
        type, criterionName, criterionID, description, name, risk, rationale

    Failure modes where risk == "high" are the "lemons" — ideas the system
    may later use to rank or eliminate dominated solutions.

    Parameters
    ----------
    solution : Solution
        The candidate solution to evaluate.
    hint : str, optional
        An optional free-text hint to guide the failure-mode generation
        (e.g. "focus on implementation complexity").  A good hint can
        surface lemons that the LLM might otherwise miss.
    """
    # Step 1: (find-criteria solution)
    criteria: list[Criterion] = find_criteria(solution)

    # Step 2: (find-if #'(lambda (p) (isa-p p '(goal barrier cause))) (gbsmcontext solution))
    purpose: Optional[Purpose] = find_purpose(solution)

    # Step 3: build prompt  (with-output-to-string block)
    prompt: str = _build_prompt(purpose, criteria, solution, hint)

    # Step 4: (askLLM prompt)
    response: str = ask_llm(prompt)

    # Step 5: (parse-llm-output response)
    failure_modes: list[dict[str, Any]] = parse_llm_output(response)

    return failure_modes
