from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from .config import config
from .pipeline import run_full_pipeline, save_output

SAMPLE_IDEAS = [
    "Create a neighborhood tool-lending library for DIY home repairs.",
    "Build a rideshare network for early-morning airport commuters.",
    "Launch a citywide compost pickup service for apartment buildings.",
    "Develop a mobile app that matches volunteers with local tutoring needs.",
    "Install solar canopies in underused parking lots to power nearby schools.",
    "Offer a subscription service for reusable packaging in restaurants.",
    "Expand on-demand bike lanes that appear during rush hour.",
    "Create micro-grants for community-run urban gardens.",
    "Deploy low-cost air quality sensors with public dashboards.",
    "Enable workplace flex-hours to reduce commuter peak load.",
    "Introduce a digital marketplace for local eldercare co-ops.",
    "Create a neighborhood-based trash collection rewards program.",
    "Design a modular pop-up retail platform for new small businesses.",
    "Build a peer-to-peer childcare swap system for parents.",
    "Launch an educational campaign that gamifies energy conservation."
]


def main() -> None:
    load_dotenv()
    output = run_full_pipeline(
        SAMPLE_IDEAS,
        problem_statement="Improve the sustainability and livability of an urban neighborhood.",
        documents=None,
    )

    output_path = Path(__file__).resolve().parent / "lemon_agent_output.json"
    save_output(output, str(output_path))

    print(f"Saved final Lemon Agent output to: {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
