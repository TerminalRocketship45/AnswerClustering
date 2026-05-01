"""
Data models for the Lemon Agent.

In the original Lisp system (Mark Klein, MIT), solutions live inside a
Goal-Barrier-Solution-Mechanism (GBSM) graph.  Each node has:
  - eid         : unique string ID, e.g. "C1", "G2"
  - name        : short label
  - description : optional longer text
  - ptype       : one of 'goal', 'barrier', 'cause', 'solution'

A Solution node is connected to one or more Purpose nodes (goal / barrier / cause)
through the gbsm_context list.  Criteria are general quality dimensions that apply
to *all* solutions (e.g. "Effective", "Affordable", "Low-Risk").

FailureMode is the structured output produced by find_failure_modes.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


PurposeType = Literal["goal", "barrier", "cause"]
RiskLevel   = Literal["high", "low"]


@dataclass
class Criterion:
    """A single evaluation dimension, e.g. 'Affordable' or 'Low-Risk'."""
    eid: str                       # Unique ID, e.g. "C1"
    name: str                      # Short label, e.g. "Affordable"
    description: Optional[str] = None   # One-sentence definition (optional)


@dataclass
class Purpose:
    """
    A node in the GBSM graph that gives a solution its reason for existing.

    ptype = 'goal'    -> we are trying to ACHIEVE something
    ptype = 'barrier' -> we are trying to OVERCOME an obstacle
    ptype = 'cause'   -> we are addressing a ROOT CAUSE of a barrier
    """
    eid: str
    name: str
    ptype: PurposeType
    description: Optional[str] = None
    parent: Optional["Purpose"] = None   # the node above this one in the GBSM tree


@dataclass
class Solution:
    """
    A candidate answer to a problem.

    gbsm_context holds all Purpose nodes in the branch that leads to this
    solution (root first).  explain_purpose() walks this list to narrate
    the chain: e.g. "We want to reduce traffic (goal).  A barrier is lack of
    public transit (barrier).  A contributing cause is low funding (cause)."
    """
    eid: str
    name: str
    description: Optional[str] = None
    gbsm_context: list[Purpose] = field(default_factory=list)


@dataclass
class FailureMode:
    """
    One failure mode returned by the Lemon Agent for a single criterion.

    risk = 'high'  -> this solution is a LEMON for this criterion
    risk = 'low'   -> this solution likely satisfies this criterion
    """
    type: str                  # always "failure" (mirrors Lisp :type field)
    criterion_name: str        # name of the criterion that could be failed
    criterion_id: str          # eid of that criterion
    description: str           # 2-3 sentence description of the failure mode
    name: str                  # 4-6 word short label
    risk: RiskLevel            # "high" or "low"
    rationale: str             # 2-3 sentence explanation of the risk rating
