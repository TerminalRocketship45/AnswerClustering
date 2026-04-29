"""
Dry-run: verifies all logic paths without calling the Gemini API.
Patches ask_llm with a canned JSON response so parse_llm_output is also tested.
"""

import sys
import json

# ── 1. imports ──────────────────────────────────────────────────────────────
from models import Purpose, Solution, Criterion
from gbsm import find_criteria, find_purpose, explain_purpose, DEFAULT_CRITERIA
from lemon_agent import _build_prompt, find_failure_modes
import llm_client

print("[ 1/5 ] imports OK")

# ── 2. GBSM object construction ──────────────────────────────────────────────
goal = Purpose(
    eid="G1", name="Reduce urban traffic congestion", ptype="goal",
    description="Lower average commute times and vehicle density in city centres.",
    parent=None,
)
barrier = Purpose(
    eid="B1", name="Insufficient public transit capacity", ptype="barrier",
    description="Current bus and rail networks cannot absorb demand from private-car commuters.",
    parent=goal,
)
cause = Purpose(
    eid="CA1", name="Low government funding for transit infrastructure", ptype="cause",
    description="Annual transit budgets have been flat for a decade while population has grown 30%.",
    parent=barrier,
)
solution = Solution(
    eid="S1",
    name="Introduce congestion pricing for private vehicles",
    description="Charge drivers a variable fee to enter the city centre during peak hours.",
    gbsm_context=[goal, barrier, cause],
)
print("[ 2/5 ] GBSM objects constructed OK")

# ── 3. find_purpose + explain_purpose ────────────────────────────────────────
purpose = find_purpose(solution)
assert purpose is not None, "find_purpose returned None"
assert purpose.ptype in {"goal", "barrier", "cause"}, f"unexpected ptype: {purpose.ptype}"

narrative = explain_purpose(purpose)
assert "GOAL" in narrative, "explain_purpose missing GOAL line"
assert "BARRIER" in narrative, "explain_purpose missing BARRIER line"
assert "CAUSE" in narrative, "explain_purpose missing CAUSE line"
print("[ 3/5 ] find_purpose / explain_purpose OK")
print("        narrative preview:", narrative.splitlines()[0])

# ── 4. _build_prompt ─────────────────────────────────────────────────────────
criteria = find_criteria(solution)
assert len(criteria) == len(DEFAULT_CRITERIA), "criteria count mismatch"

prompt = _build_prompt(purpose, criteria, solution, hint="focus on equity")
assert "criterionName" in prompt, "JSON schema missing from prompt"
assert "Introduce congestion pricing" in prompt, "solution name missing from prompt"
assert "equity" in prompt, "hint missing from prompt"
print("[ 4/5 ] _build_prompt OK  (prompt length:", len(prompt), "chars)")

# ── 5. parse_llm_output with canned response ─────────────────────────────────
CANNED = json.dumps([
    {
        "type": "failure",
        "criterionName": "Affordable",
        "criterionID": "C2",
        "description": "Low-income drivers may be disproportionately burdened.",
        "name": "Regressive Cost Burden on Poor",
        "risk": "high",
        "rationale": "Flat fees hit low-income workers hardest since they cannot afford alternatives."
    },
    {
        "type": "failure",
        "criterionName": "Effective",
        "criterionID": "C1",
        "description": "Drivers may shift routes rather than reduce trips.",
        "name": "Route Displacement Not Reduction",
        "risk": "low",
        "rationale": "Evidence from London and Stockholm shows net vehicle-km reduction within the zone."
    }
])

# Monkey-patch at the lemon_agent module level so the local reference is replaced.
# (lemon_agent does `from llm_client import ask_llm`, so patching llm_client
#  directly has no effect on the already-bound name inside lemon_agent.)
import lemon_agent as _la  # noqa: E402 — must come after the patch setup
original_ask = _la.ask_llm
_la.ask_llm = lambda prompt, **kw: f"```json\n{CANNED}\n```"

result = find_failure_modes(solution, hint="equity")

_la.ask_llm = original_ask   # restore

assert len(result) == 2, f"expected 2 items, got {len(result)}"
assert result[0]["risk"] == "high"
assert result[1]["risk"] == "low"
print("[ 5/5 ] parse_llm_output + find_failure_modes pipeline OK")

print()
print("=" * 60)
print("  DRY RUN PASSED — safe to run with real API key")
print("=" * 60)
