"""
Example runner — demonstrates find_failure_modes end-to-end.

Scenario (adapted from Mark Klein's GBSM examples):
  GOAL     : Reduce urban traffic congestion
  BARRIER  : Insufficient public transit capacity
  CAUSE    : Low government funding for transit infrastructure
  SOLUTION : Introduce congestion pricing for private vehicles

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_example.py

The script prints each failure mode and highlights the lemons (high risk).
"""

import json
from models import Purpose, Solution
from lemon_agent import find_failure_modes


def main() -> None:
    # --- Build a small GBSM branch ---

    goal = Purpose(
        eid="G1",
        name="Reduce urban traffic congestion",
        ptype="goal",
        description="Lower average commute times and vehicle density in city centres.",
        parent=None,
    )

    barrier = Purpose(
        eid="B1",
        name="Insufficient public transit capacity",
        ptype="barrier",
        description="Current bus and rail networks cannot absorb demand from private-car commuters.",
        parent=goal,
    )

    cause = Purpose(
        eid="CA1",
        name="Low government funding for transit infrastructure",
        ptype="cause",
        description="Annual transit budgets have been flat for a decade while population has grown 30 %.",
        parent=barrier,
    )

    solution = Solution(
        eid="S1",
        name="Introduce congestion pricing for private vehicles",
        description=(
            "Charge drivers a variable fee to enter the city centre during peak hours, "
            "using revenue to subsidise public transit expansion."
        ),
        # The GBSM context lists the branch from root to the immediate purpose.
        # find_purpose() will pick the first goal/barrier/cause it finds.
        gbsm_context=[goal, barrier, cause],
    )

    # --- Run the Lemon Agent ---
    print("Running Lemon Agent …\n")
    failure_modes = find_failure_modes(
        solution,
        hint="Consider political feasibility and equity impacts on low-income drivers.",
    )

    # --- Display results ---
    lemons = [fm for fm in failure_modes if fm.get("risk", "").lower() == "high"]
    safe   = [fm for fm in failure_modes if fm.get("risk", "").lower() != "high"]

    print(f"Found {len(failure_modes)} failure modes  "
          f"({len(lemons)} lemons, {len(safe)} low-risk)\n")
    print("=" * 70)

    for fm in failure_modes:
        risk_label = "LEMON " if fm.get("risk", "").lower() == "high" else "      "
        print(f"{risk_label}[{fm.get('criterionID')}] {fm.get('name')}")
        print(f"  Criterion : {fm.get('criterionName')}")
        print(f"  Risk      : {fm.get('risk', '').upper()}")
        print(f"  Description: {fm.get('description')}")
        print(f"  Rationale : {fm.get('rationale')}")
        print()

    # Also dump raw JSON for downstream use
    print("--- Raw JSON ---")
    print(json.dumps(failure_modes, indent=2))


if __name__ == "__main__":
    main()
