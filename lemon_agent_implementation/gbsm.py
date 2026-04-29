"""
GBSM (Goal-Barrier-Solution-Mechanism) helper functions.

Mirrors the Lisp helpers referenced in find-failure-modes:

  (find-criteria solution)
      -> returns the list of Criterion objects that apply to this solution.
         In the original system these are general quality dimensions (effective,
         affordable, low-risk, …).  Here we return a default set so the agent
         can run without a full knowledge base; you can replace this with a
         database lookup or an LLM-generated criteria list.

  (find-if #'(lambda (p) (isa-p p '(goal barrier cause))) (gbsmcontext solution))
      -> walks the gbsm_context of the solution and returns the *first* node
         whose ptype is one of goal / barrier / cause.

  (explain-purpose purpose)
      -> builds a natural-language narrative of the GBSM branch from root to
         the purpose node, giving the LLM the "why" behind the solution.
"""

from __future__ import annotations
from typing import Optional
from models import Criterion, Purpose, Solution


# ---------------------------------------------------------------------------
# Default criteria
# ---------------------------------------------------------------------------

DEFAULT_CRITERIA: list[Criterion] = [
    Criterion(eid="C1", name="Effective",     description="The solution actually achieves the intended goal."),
    Criterion(eid="C2", name="Affordable",    description="The solution can be implemented within reasonable cost constraints."),
    Criterion(eid="C3", name="Feasible",      description="The solution can realistically be implemented with available resources and technology."),
    Criterion(eid="C4", name="Low-Risk",      description="The solution does not introduce significant new risks or side-effects."),
    Criterion(eid="C5", name="Timely",        description="The solution can be delivered within an acceptable timeframe."),
    Criterion(eid="C6", name="Sustainable",   description="The solution can be maintained and scaled over time."),
]


def find_criteria(solution: Solution) -> list[Criterion]:
    """
    Return the list of criteria that apply to this solution.

    In a full GBSM knowledge base this would query a database.  For now we
    return the shared default set; replace with your own lookup as needed.
    """
    # TODO: extend with per-solution or per-domain overrides
    return DEFAULT_CRITERIA


def find_purpose(solution: Solution) -> Optional[Purpose]:
    """
    Lisp: (find-if #'(lambda (p) (isa-p p '(goal barrier cause))) (gbsmcontext solution))

    Returns the *deepest* (most specific) Purpose node in the solution's GBSM
    context whose ptype is one of: 'goal', 'barrier', 'cause'.

    Returning the deepest node lets explain_purpose walk the full parent chain
    back to the root, producing the complete motivational narrative.
    In the Lisp system gbsmcontext was ordered deepest-first so find-if
    naturally found the deepest node; here we scan the whole list instead.
    """
    valid_types = {"goal", "barrier", "cause"}
    result: Optional[Purpose] = None
    for node in solution.gbsm_context:
        if node.ptype in valid_types:
            result = node   # keep overwriting — last match is deepest
    return result


def explain_purpose(purpose: Optional[Purpose]) -> str:
    """
    Lisp: (explain-purpose purpose)

    Walks from the root of the GBSM branch down to `purpose` and produces a
    narrative that gives the LLM the full motivational chain.

    Example output:
        "Context:
         GOAL: Reduce urban traffic congestion.
         BARRIER: Lack of affordable public transit.
         CAUSE: Insufficient government funding for transit infrastructure."
    """
    if purpose is None:
        return "Context: (no purpose context available)"

    # Collect ancestors by walking the parent chain
    chain: list[Purpose] = []
    node: Optional[Purpose] = purpose
    while node is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()   # root first

    lines = ["Context:"]
    for p in chain:
        label = p.ptype.upper()
        desc  = f" — {p.description}" if p.description else ""
        lines.append(f"  {label}: {p.name}{desc}")
    return "\n".join(lines)
