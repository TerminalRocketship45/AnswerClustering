from __future__ import annotations

from typing import List, Optional

from .models import Criterion

SYSTEM_PROMPT = (
    "You are a rigorous decision-analysis expert. Your job is to design and apply "
    "structured evaluation criteria that separate strong ideas from weak ideas. "
    "Respond only with valid JSON for the requested output format."
)


def build_criteria_prompt(
    ideas: List[str],
    k_criteria: int,
    problem_statement: Optional[str] = None,
) -> str:
    prompt_lines = [
        "Given the sample candidate ideas below, identify the most discriminative evaluation criteria for this set.",
        "Discriminative criteria are the dimensions that best separate good ideas from weak ones in this domain.",
        "Avoid redundant or overlapping criteria. Do not invent a problem statement if none is provided.",
    ]

    if problem_statement:
        prompt_lines.append(f"Problem statement: {problem_statement}")

    prompt_lines.append("Sample ideas:")
    for index, idea in enumerate(ideas, start=1):
        prompt_lines.append(f"{index}. {idea}")

    prompt_lines.extend([
        "\nReturn exactly a JSON array with this schema:",
        "[",
        "  {\n    \"id\": \"C1\",\n    \"name\": \"Short criterion name\",\n    \"description\": \"One sentence definition of the criterion.\"\n  },",
        "  ...",
        "]",
        f"Return exactly {k_criteria} criteria.",
    ])
    return "\n".join(prompt_lines)


def build_failure_modes_prompt(
    ideas: List[str],
    criteria: List[Criterion],
    failure_modes_per_criterion: int,
    retrieved_context: Optional[str] = None,
    hint: Optional[str] = None,
    style: str = "statement",
) -> str:
    prompt_lines = [
        "You are a rigorous failure-mode analyst.",
        "For each candidate solution below, identify distinct failure modes tied to the evaluation criteria.",
        "Use the solution text and any provided context to decide how the solution could fail each criterion.",
        "If GBSM context is provided, include it explicitly in your analysis of goals, barriers, or causes.",
        "Return only valid JSON in the format described. Do not include any explanatory text outside the JSON array.",
        "",
    ]

    if retrieved_context:
        prompt_lines.extend([
            "Context:",
            retrieved_context,
            "",
        ])

    prompt_lines.append("Criteria:")
    for criterion in criteria:
        prompt_lines.append(f"- {criterion.eid}: {criterion.name} — {criterion.description}")

    prompt_lines.append("")
    prompt_lines.append("Solutions:")
    for index, idea in enumerate(ideas, start=1):
        prompt_lines.append(f"{index}. {idea}")

    if hint:
        prompt_lines.extend([
            "",
            f"Hint: {hint}",
        ])

    if style == "question":
        prompt_lines.append(
            "Express each failure mode as a concrete evaluation question that captures how the solution could fail the criterion."
        )
    else:
        prompt_lines.append(
            "Express each failure mode as a concise failure statement describing how the solution can fail the criterion."
        )

    prompt_lines.extend([
        "",
        "Output schema:",
        "[",
        "  {",
        "    \"type\": \"failure\",",
        "    \"solutionName\": \"Name of the candidate solution\",",
        "    \"criterionID\": \"C1\",",
        "    \"criterionName\": \"Short criterion name\",",
        "    \"name\": \"Short failure mode name\",",
        "    \"description\": \"Two to three sentence description of the failure mode.\",
        "    \"risk\": \"high|medium|low\",",
        "    \"rationale\": \"Two to three sentence explanation of the risk rating.\",
        "  },",
        "  ...",
        "]",
        f"Generate exactly {failure_modes_per_criterion} failure modes for each criterion and each solution.",
        "Use high/medium/low risk ratings.",
    ])
    return "\n".join(prompt_lines)


def build_batch_evaluation_prompt(
    ideas: List[str],
    failure_modes: List[dict],
    rating_levels: int,
    retrieved_context: Optional[str] = None,
) -> str:
    rating_scale = "yes/no" if rating_levels == 2 else "high/medium/low"
    prompt_lines = [
        "You will evaluate a batch of ideas against a shared set of failure questions.",
        "Use the entire batch to keep your calibration consistent, but rate each idea individually.",
        "Do not invent details beyond the idea text and any provided context.",
        "For each idea and question, provide a short reasoning statement, then the rating.",
        "Return only valid JSON with the structure below.",
        "",
    ]

    if retrieved_context:
        prompt_lines.extend([
            "Retrieved context:",
            retrieved_context,
            "",
        ])

    prompt_lines.append("Ideas:")
    for index, idea in enumerate(ideas, start=1):
        prompt_lines.append(f"{index}. {idea}")

    prompt_lines.append("")
    prompt_lines.append("Failure modes:")
    for index, failure_mode in enumerate(failure_modes, start=1):
        prompt_lines.append(
            f"{index}. [{failure_mode['criterionID']}] {failure_mode['criterionName']} — {failure_mode['description']}"
        )

    prompt_lines.extend([
        "",
        "Output schema:",
        "[",
        "  {",
        "    \"idea\": \"Idea text\",",
        "    \"evaluations\": [",
        "      {",
        "        \"criterionID\": \"C1\",",
        "        \"criterionName\": \"Cost\",",
        "        \"failureModes\": [",
        "          {\n            \"failureMode\": \"Does this solution require upfront infrastructure investment exceeding $10M?\",\n            \"rating\": \"yes\",\n            \"reasoning\": \"Short chain-of-thought rationale.\"\n          },",
        "          ...",
        "        ]",
        "      },",
        "      ...",
        "    ]",
        "  },",
        "  ...",
        "]",
        f"Use the rating scale: {rating_scale}.",
        "Keep the output JSON clean and free of explanatory text outside the JSON structure.",
    ])
    return "\n".join(prompt_lines)
