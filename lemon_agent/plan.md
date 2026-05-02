# Lemon Agent Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Lemon Agent so all ideas are rated against a shared set of universal failure modes, producing aligned, comparable vectors for valid Pareto dominance filtering and downstream clustering.

**Architecture:** Three clean stages — (1) criteria generation (unchanged), (2) universal failure mode generation with no idea text, optional GBSM purpose context injected as preamble, (3) batch rating of ~10 ideas at a time against the shared failure modes, with rationale as chain-of-thought. The changes are surgical: models, prompts, and the pipeline's two new functions are modified; RAG, LLM client, Pareto logic, and config structure are untouched.

**Tech Stack:** Python 3.11+, google-generativeai (Gemini), pytest

---

## File Map

| File | Change |
|---|---|
| `lemon_agent/models.py` | Simplify `FailureMode`, remove `CriterionEvaluation`/`FailureModeResult`, add `FailureModeRating`, update `IdeaEvaluation` |
| `lemon_agent/prompt_builder.py` | Redesign `build_failure_modes_prompt`, add `build_batch_rating_prompt`, remove `build_batch_evaluation_prompt` |
| `lemon_agent/pipeline.py` | Add `_validate_rating_levels`, redesign `generate_failure_modes`, add `rate_ideas_batch`, remove `evaluate_batch`, update `_build_failure_mode_vector` + `assign_lemon_labels` + `run_full_pipeline` |
| `tests/test_models.py` | New — tests for `FailureModeRating`, updated `IdeaEvaluation.to_dict()` |
| `tests/test_prompt_builder.py` | New — tests for both prompt functions |
| `tests/test_pipeline.py` | New — tests for pure functions: `_validate_rating_levels`, `_build_failure_mode_vector`, `_is_pareto_dominated` |

---

## Task 1: Update `models.py`

**Files:**
- Modify: `lemon_agent/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lemon_agent.models import FailureMode, FailureModeRating, IdeaEvaluation


def test_failure_mode_has_no_solution_name():
    fm = FailureMode(
        type="failure",
        criterionID="C1",
        criterionName="Cost",
        name="Budget overrun",
        description="Costs exceed available funding.",
    )
    assert not hasattr(fm, "solutionName")
    assert not hasattr(fm, "risk")
    assert not hasattr(fm, "rationale")


def test_failure_mode_rating_fields():
    r = FailureModeRating(risk="high", rationale="This idea requires expensive infrastructure.")
    assert r.risk == "high"
    assert r.rationale == "This idea requires expensive infrastructure."


def test_idea_evaluation_to_dict_uses_ratings():
    eval_ = IdeaEvaluation(
        idea="Solar canopies on parking lots",
        ratings=[
            FailureModeRating(risk="high", rationale="High upfront cost."),
            FailureModeRating(risk="low", rationale="Strong political backing."),
        ],
        failure_mode_vector=[1.0, -1.0],
        is_lemon=True,
    )
    d = eval_.to_dict()
    assert d["idea"] == "Solar canopies on parking lots"
    assert len(d["ratings"]) == 2
    assert d["ratings"][0] == {"risk": "high", "rationale": "High upfront cost."}
    assert d["ratings"][1] == {"risk": "low", "rationale": "Strong political backing."}
    assert d["failure_mode_vector"] == [1.0, -1.0]
    assert d["is_lemon"] is True
    assert "evaluations" not in d
    assert "failure_modes" not in d


def test_idea_evaluation_defaults():
    eval_ = IdeaEvaluation(idea="Test idea")
    assert eval_.ratings == []
    assert eval_.failure_mode_vector == []
    assert eval_.is_lemon is False
```

- [ ] **Step 2: Run tests — verify they fail**

```
cd C:\Users\rohan\Downloads\ML\AnswerClustering
pytest tests/test_models.py -v
```

Expected: errors because `FailureModeRating` doesn't exist and `FailureMode` still has `solutionName`.

- [ ] **Step 3: Rewrite `lemon_agent/models.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, List

RatingValue = Literal["high", "medium", "low"]


@dataclass
class Criterion:
    eid: str
    name: str
    description: str


@dataclass
class FailureMode:
    type: str
    criterionID: str
    criterionName: str
    name: str
    description: str


@dataclass
class FailureModeRating:
    risk: RatingValue
    rationale: str


@dataclass
class IdeaEvaluation:
    idea: str
    ratings: List[FailureModeRating] = field(default_factory=list)
    failure_mode_vector: List[float] = field(default_factory=list)
    is_lemon: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea,
            "ratings": [
                {"risk": r.risk, "rationale": r.rationale}
                for r in self.ratings
            ],
            "failure_mode_vector": self.failure_mode_vector,
            "is_lemon": self.is_lemon,
        }
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_models.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add lemon_agent/models.py tests/test_models.py
git commit -m "refactor(models): simplify FailureMode, add FailureModeRating, update IdeaEvaluation"
```

---

## Task 2: Redesign `prompt_builder.py`

**Files:**
- Modify: `lemon_agent/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompt_builder.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lemon_agent.models import Criterion, FailureMode
from lemon_agent.prompt_builder import build_failure_modes_prompt, build_batch_rating_prompt


def _make_criteria():
    return [
        Criterion(eid="C1", name="Cost", description="Must be affordable."),
        Criterion(eid="C2", name="Feasibility", description="Must be technically achievable."),
    ]


def _make_shared_fms():
    return [
        FailureMode(type="failure", criterionID="C1", criterionName="Cost",
                    name="Budget overrun", description="Costs exceed available funding."),
        FailureMode(type="failure", criterionID="C2", criterionName="Feasibility",
                    name="Technical complexity", description="Implementation requires unavailable expertise."),
    ]


# build_failure_modes_prompt tests

def test_failure_modes_prompt_has_no_idea_text():
    prompt = build_failure_modes_prompt(_make_criteria(), failure_modes_per_criterion=2)
    # must not contain any instruction to evaluate specific ideas
    assert "idea 1" not in prompt.lower()
    assert "solution 1" not in prompt.lower()

def test_failure_modes_prompt_contains_criteria():
    prompt = build_failure_modes_prompt(_make_criteria(), failure_modes_per_criterion=2)
    assert "C1" in prompt
    assert "Cost" in prompt
    assert "C2" in prompt
    assert "Feasibility" in prompt

def test_failure_modes_prompt_no_risk_in_schema():
    prompt = build_failure_modes_prompt(_make_criteria(), failure_modes_per_criterion=2)
    # schema should not ask the LLM to output a risk rating
    assert '"risk"' not in prompt

def test_failure_modes_prompt_specifies_count():
    prompt = build_failure_modes_prompt(_make_criteria(), failure_modes_per_criterion=3)
    assert "3" in prompt

def test_failure_modes_prompt_with_purpose():
    purpose = "We are trying to overcome the barrier of urban traffic congestion."
    prompt = build_failure_modes_prompt(_make_criteria(), failure_modes_per_criterion=2, purpose=purpose)
    assert "urban traffic congestion" in prompt

def test_failure_modes_prompt_with_context():
    prompt = build_failure_modes_prompt(
        _make_criteria(), failure_modes_per_criterion=2,
        retrieved_context="Congestion pricing was trialled in Stockholm in 2006."
    )
    assert "Stockholm" in prompt


# build_batch_rating_prompt tests

def test_batch_rating_prompt_two_levels():
    prompt = build_batch_rating_prompt(
        ideas=["Idea A", "Idea B"],
        failure_modes=_make_shared_fms(),
        rating_levels=2,
    )
    assert '"high" or "low"' in prompt
    assert "medium" not in prompt

def test_batch_rating_prompt_three_levels():
    prompt = build_batch_rating_prompt(
        ideas=["Idea A"],
        failure_modes=_make_shared_fms(),
        rating_levels=3,
    )
    assert '"high", "medium", or "low"' in prompt

def test_batch_rating_prompt_contains_all_ideas():
    ideas = ["Solar canopies", "Bike lanes", "Compost pickup"]
    prompt = build_batch_rating_prompt(ideas, _make_shared_fms(), rating_levels=3)
    for idea in ideas:
        assert idea in prompt

def test_batch_rating_prompt_contains_all_failure_modes():
    prompt = build_batch_rating_prompt(["Idea A"], _make_shared_fms(), rating_levels=3)
    assert "Budget overrun" in prompt
    assert "Technical complexity" in prompt

def test_batch_rating_prompt_with_purpose():
    purpose = "We are trying to achieve the goal of reducing carbon emissions."
    prompt = build_batch_rating_prompt(
        ["Idea A"], _make_shared_fms(), rating_levels=3, purpose=purpose
    )
    assert "carbon emissions" in prompt

def test_batch_rating_prompt_rationale_instruction():
    prompt = build_batch_rating_prompt(["Idea A"], _make_shared_fms(), rating_levels=3)
    assert "rationale" in prompt
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_prompt_builder.py -v
```

Expected: failures because `build_batch_rating_prompt` doesn't exist and `build_failure_modes_prompt` still accepts ideas.

- [ ] **Step 3: Rewrite `lemon_agent/prompt_builder.py`**

```python
from __future__ import annotations

from typing import List, Optional

from .models import Criterion, FailureMode

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
    criteria: List[Criterion],
    failure_modes_per_criterion: int,
    purpose: Optional[str] = None,
    retrieved_context: Optional[str] = None,
) -> str:
    prompt_lines = []

    if purpose:
        prompt_lines.extend([purpose, ""])

    prompt_lines.extend([
        "I want you to enumerate, for each of the criteria below, the ways that a solution could fail to meet that criterion.",
        "These failure modes should be domain-level — they describe how any solution in this problem space could fail, independent of any specific idea.",
        "Return only valid JSON. Do not include any explanatory text outside the JSON array.",
        "",
    ])

    if retrieved_context:
        prompt_lines.extend([
            "Context:",
            retrieved_context,
            "",
        ])

    prompt_lines.append("Criteria:")
    for criterion in criteria:
        prompt_lines.append(f"- {criterion.eid}: {criterion.name} — {criterion.description}")

    prompt_lines.extend([
        "",
        "Output schema:",
        "[",
        "  {",
        '    "type": "failure",',
        '    "criterionID": "C1",',
        '    "criterionName": "Short criterion name",',
        '    "name": "Short failure mode name (4-6 words)",',
        '    "description": "Two to three sentence description of this failure mode."',
        "  },",
        "  ...",
        "]",
        f"Generate exactly {failure_modes_per_criterion} failure modes per criterion.",
        "Do not include risk ratings — those are assigned per idea in a separate step.",
    ])
    return "\n".join(prompt_lines)


def build_batch_rating_prompt(
    ideas: List[str],
    failure_modes: List[FailureMode],
    rating_levels: int,
    purpose: Optional[str] = None,
    retrieved_context: Optional[str] = None,
) -> str:
    if rating_levels == 2:
        allowed_values = '"high" or "low"'
    else:
        allowed_values = '"high", "medium", or "low"'

    prompt_lines = [
        "You will rate a batch of ideas against a shared set of failure modes.",
        "Use the full batch to calibrate your ratings comparatively, but rate each idea independently in the output.",
        "For each idea and each failure mode, provide a 2-3 sentence rationale explaining why this failure is or is not a concern for this specific idea, then give a risk rating.",
        "Return only valid JSON with the structure below.",
        "",
    ]

    if purpose:
        prompt_lines.extend([purpose, ""])

    if retrieved_context:
        prompt_lines.extend([
            "Context:",
            retrieved_context,
            "",
        ])

    prompt_lines.append("Ideas:")
    for index, idea in enumerate(ideas, start=1):
        prompt_lines.append(f"{index}. {idea}")

    prompt_lines.append("")
    prompt_lines.append("Failure modes (shared across all ideas, rate each idea against every one):")
    for index, fm in enumerate(failure_modes, start=1):
        prompt_lines.append(
            f"{index}. [{fm.criterionID}] {fm.criterionName} — {fm.name}: {fm.description}"
        )

    prompt_lines.extend([
        "",
        "Output schema:",
        "[",
        "  {",
        '    "idea": "Idea text",',
        '    "ratings": [',
        "      {",
        '        "risk": "...",',
        '        "rationale": "2-3 sentence explanation for this specific idea."',
        "      },",
        "      ... (one entry per failure mode, in the same order as the list above)",
        "    ]",
        "  },",
        "  ...",
        "]",
        f"Use only these risk values: {allowed_values}.",
        "Output exactly one ratings entry per failure mode per idea, in the same order as the failure modes list above.",
        "Keep the output JSON clean and free of explanatory text outside the JSON structure.",
    ])
    return "\n".join(prompt_lines)
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_prompt_builder.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```
git add lemon_agent/prompt_builder.py tests/test_prompt_builder.py
git commit -m "refactor(prompts): universal FM prompt, batch rating prompt, remove evaluate_batch prompt"
```

---

## Task 3: Update pure pipeline functions + add validation

**Files:**
- Modify: `lemon_agent/pipeline.py` (only `_validate_rating_levels`, `_build_failure_mode_vector`, `assign_lemon_labels`)
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from lemon_agent.models import FailureModeRating, IdeaEvaluation, Criterion
from lemon_agent.pipeline import (
    _validate_rating_levels,
    _build_failure_mode_vector,
    _is_pareto_dominated,
    assign_lemon_labels,
)


# _validate_rating_levels

def test_validate_rating_levels_accepts_2():
    _validate_rating_levels(2)  # no error


def test_validate_rating_levels_accepts_3():
    _validate_rating_levels(3)  # no error


def test_validate_rating_levels_rejects_1():
    with pytest.raises(ValueError, match="must be 2 or 3"):
        _validate_rating_levels(1)


def test_validate_rating_levels_rejects_4():
    with pytest.raises(ValueError, match="must be 2 or 3"):
        _validate_rating_levels(4)


# _build_failure_mode_vector

def test_vector_high_medium_low():
    ratings = [
        FailureModeRating(risk="high", rationale=""),
        FailureModeRating(risk="medium", rationale=""),
        FailureModeRating(risk="low", rationale=""),
    ]
    assert _build_failure_mode_vector(ratings, 3) == [1.0, 0.0, -1.0]


def test_vector_two_level():
    ratings = [
        FailureModeRating(risk="high", rationale=""),
        FailureModeRating(risk="low", rationale=""),
    ]
    assert _build_failure_mode_vector(ratings, 2) == [1.0, -1.0]


def test_vector_padding():
    ratings = [FailureModeRating(risk="high", rationale="")]
    vector = _build_failure_mode_vector(ratings, 3)
    assert vector == [1.0, 0.0, 0.0]


def test_vector_truncation():
    ratings = [
        FailureModeRating(risk="high", rationale=""),
        FailureModeRating(risk="medium", rationale=""),
        FailureModeRating(risk="low", rationale=""),
    ]
    assert _build_failure_mode_vector(ratings, 2) == [1.0, 0.0]


# _is_pareto_dominated (unchanged logic — regression test)

def test_pareto_dominated():
    target = [1.0, 1.0]
    candidate = [0.0, 1.0]  # better on first, equal on second
    assert _is_pareto_dominated(target, candidate) is True


def test_pareto_not_dominated_equal():
    target = [1.0, 1.0]
    candidate = [1.0, 1.0]  # equal, not strictly better anywhere
    assert _is_pareto_dominated(target, candidate) is False


def test_pareto_not_dominated_worse():
    target = [0.0, 1.0]
    candidate = [1.0, 0.0]  # neither dominates the other
    assert _is_pareto_dominated(target, candidate) is False


# assign_lemon_labels (integration with new ratings field)

def test_assign_lemon_labels_identifies_lemon():
    # idea A has all high risk, idea B has all low risk — A should be lemon
    idea_a = IdeaEvaluation(
        idea="Idea A",
        ratings=[
            FailureModeRating(risk="high", rationale=""),
            FailureModeRating(risk="high", rationale=""),
        ],
    )
    idea_b = IdeaEvaluation(
        idea="Idea B",
        ratings=[
            FailureModeRating(risk="low", rationale=""),
            FailureModeRating(risk="low", rationale=""),
        ],
    )
    criteria = [Criterion(eid="C1", name="Cost", description="")]
    results = assign_lemon_labels([idea_a, idea_b], criteria)
    assert results[0].is_lemon is True
    assert results[1].is_lemon is False


def test_assign_lemon_labels_no_lemon_when_incomparable():
    idea_a = IdeaEvaluation(
        idea="Idea A",
        ratings=[
            FailureModeRating(risk="high", rationale=""),
            FailureModeRating(risk="low", rationale=""),
        ],
    )
    idea_b = IdeaEvaluation(
        idea="Idea B",
        ratings=[
            FailureModeRating(risk="low", rationale=""),
            FailureModeRating(risk="high", rationale=""),
        ],
    )
    criteria = [Criterion(eid="C1", name="Cost", description="")]
    results = assign_lemon_labels([idea_a, idea_b], criteria)
    assert results[0].is_lemon is False
    assert results[1].is_lemon is False
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_pipeline.py -v
```

Expected: failures on `_validate_rating_levels` (doesn't exist) and `_build_failure_mode_vector` (wrong input type).

- [ ] **Step 3: Add `_validate_rating_levels` to `pipeline.py`**

Add this function near the top of `lemon_agent/pipeline.py` (after the imports):

```python
def _validate_rating_levels(rating_levels: int) -> None:
    if rating_levels not in (2, 3):
        raise ValueError(
            f"config['rating_levels'] must be 2 or 3, got {rating_levels}. "
            "Update lemon_agent/config.py to use a valid value."
        )
```

- [ ] **Step 4: Update `_build_failure_mode_vector` in `pipeline.py`**

Replace the existing function:

```python
def _build_failure_mode_vector(
    ratings: List[FailureModeRating],
    expected_length: int,
) -> List[float]:
    risk_map = {
        "high": 1.0,
        "medium": 0.0,
        "low": -1.0,
    }
    vector = [risk_map.get(r.risk, 0.0) for r in ratings]
    if len(vector) < expected_length:
        vector.extend([0.0] * (expected_length - len(vector)))
    return vector[:expected_length]
```

- [ ] **Step 5: Update `assign_lemon_labels` in `pipeline.py` to use `ratings`**

Replace the existing function:

```python
def assign_lemon_labels(
    evaluations: List[IdeaEvaluation],
    criteria: List[Criterion],
) -> List[IdeaEvaluation]:
    expected_len = len(criteria) * config["failure_modes_per_criterion"]
    vectors = [
        _build_failure_mode_vector(eval_item.ratings, expected_len)
        for eval_item in evaluations
    ]

    for idea_eval, vector in zip(evaluations, vectors):
        idea_eval.failure_mode_vector = vector

    for idx, idea_eval in enumerate(evaluations):
        computed = False
        target_vector = vectors[idx]
        for jdx, other_vector in enumerate(vectors):
            if idx == jdx:
                continue
            if _is_pareto_dominated(target_vector, other_vector):
                idea_eval.is_lemon = True
                computed = True
                break
        if not computed:
            idea_eval.is_lemon = False

    return evaluations
```

- [ ] **Step 6: Update imports in `pipeline.py`**

In the imports at the top of `pipeline.py`, replace:

```python
from .models import (
    Criterion,
    FailureMode,
    FailureModeResult,
    CriterionEvaluation,
    IdeaEvaluation,
)
from .prompt_builder import (
    SYSTEM_PROMPT,
    build_batch_evaluation_prompt,
    build_criteria_prompt,
    build_failure_modes_prompt,
)
```

With:

```python
from .models import (
    Criterion,
    FailureMode,
    FailureModeRating,
    IdeaEvaluation,
)
from .prompt_builder import (
    SYSTEM_PROMPT,
    build_batch_rating_prompt,
    build_criteria_prompt,
    build_failure_modes_prompt,
)
```

- [ ] **Step 7: Run tests — verify they pass**

```
pytest tests/test_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```
git add lemon_agent/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): update vector builder to use FailureModeRating, add rating_levels validation"
```

---

## Task 4: Redesign `generate_failure_modes` in `pipeline.py`

**Files:**
- Modify: `lemon_agent/pipeline.py`

- [ ] **Step 1: Replace `generate_failure_modes`**

Replace the existing `generate_failure_modes` function with:

```python
def generate_failure_modes(
    criteria: List[Criterion],
    failure_modes_per_criterion: int,
    retrieved_context: Optional[str] = None,
    purpose: Optional[str] = None,
) -> List[FailureMode]:
    prompt = build_failure_modes_prompt(
        criteria,
        failure_modes_per_criterion,
        purpose=purpose,
        retrieved_context=retrieved_context,
    )
    raw = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    failure_modes: List[FailureMode] = []
    for item in raw:
        failure_modes.append(
            FailureMode(
                type=item.get("type", "failure"),
                criterionID=item.get("criterionID", ""),
                criterionName=item.get("criterionName", ""),
                name=item.get("name", ""),
                description=item.get("description", ""),
            )
        )
    return failure_modes
```

- [ ] **Step 2: Remove now-unused helper `_rating_to_score`**

Delete the `_rating_to_score` function entirely — it was only used by `evaluate_batch`, which is being removed.

- [ ] **Step 3: Run existing tests — verify nothing broke**

```
pytest tests/ -v
```

Expected: all tests pass (no regressions).

- [ ] **Step 4: Commit**

```
git add lemon_agent/pipeline.py
git commit -m "refactor(pipeline): generate_failure_modes now produces universal shared FMs without idea text"
```

---

## Task 5: Add `rate_ideas_batch`, remove `evaluate_batch`

**Files:**
- Modify: `lemon_agent/pipeline.py`

- [ ] **Step 1: Add `rate_ideas_batch` after `generate_failure_modes`**

```python
def rate_ideas_batch(
    ideas: List[str],
    shared_failure_modes: List[FailureMode],
    retrieved_context: Optional[str] = None,
    purpose: Optional[str] = None,
) -> List[IdeaEvaluation]:
    prompt = build_batch_rating_prompt(
        ideas,
        shared_failure_modes,
        rating_levels=config["rating_levels"],
        purpose=purpose,
        retrieved_context=retrieved_context,
    )
    raw = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    results: List[IdeaEvaluation] = []
    for idea_result in raw:
        idea_text = idea_result.get("idea", "")
        ratings: List[FailureModeRating] = []
        for rating_item in idea_result.get("ratings", []):
            ratings.append(
                FailureModeRating(
                    risk=str(rating_item.get("risk", "low")).strip().lower(),
                    rationale=rating_item.get("rationale", ""),
                )
            )
        results.append(IdeaEvaluation(idea=idea_text, ratings=ratings))
    return results
```

- [ ] **Step 2: Delete `evaluate_batch`**

Remove the entire `evaluate_batch` function from `pipeline.py`.

- [ ] **Step 3: Run all tests — verify nothing broke**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```
git add lemon_agent/pipeline.py
git commit -m "feat(pipeline): add rate_ideas_batch, remove evaluate_batch"
```

---

## Task 6: Update `run_full_pipeline`

**Files:**
- Modify: `lemon_agent/pipeline.py`

- [ ] **Step 1: Replace `run_full_pipeline`**

Replace the existing function with:

```python
def run_full_pipeline(
    ideas: List[str],
    problem_statement: Optional[str] = None,
    purpose: Optional[str] = None,
    documents: Optional[List[str]] = None,
    gbsm_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _validate_rating_levels(config["rating_levels"])

    sampled_ideas = sample_ideas(ideas, config["criteria_sample_size"])
    criteria = generate_criteria(sampled_ideas, config["k_criteria"], problem_statement)

    retrieved_context = ""
    if config["rag_enabled"]:
        documents = documents or []
        if documents:
            document_store = build_document_store(documents, model_name=config["embeddings_model"])
            retrieved_context = retrieve_context(
                problem_statement or "",
                document_store,
                model_name=config["embeddings_model"],
                top_k=config["rag_top_k"],
            )
        else:
            retrieved_context = search_fallback(problem_statement or "", model_name=config["model"])

    if gbsm_context and retrieved_context:
        fm_context = f"{gbsm_context}\n\n{retrieved_context}"
    else:
        fm_context = gbsm_context or retrieved_context or None

    # Stage 2: universal failure modes — no idea text
    shared_failure_modes = generate_failure_modes(
        criteria,
        config["failure_modes_per_criterion"],
        retrieved_context=fm_context,
        purpose=purpose,
    )

    # Stage 3: batch rating — all ideas rated against shared failure modes
    batches: List[List[str]] = []
    for i in range(0, len(ideas), config["batch_size"]):
        batches.append(ideas[i : i + config["batch_size"]])

    all_evaluations: List[IdeaEvaluation] = []
    for batch in batches:
        batch_context = retrieved_context if config["rag_enabled"] else None
        batch_results = rate_ideas_batch(
            batch,
            shared_failure_modes,
            retrieved_context=batch_context,
            purpose=purpose,
        )
        all_evaluations.extend(batch_results)

    final_evaluations = assign_lemon_labels(all_evaluations, criteria)
    return [item.to_dict() for item in final_evaluations]
```

- [ ] **Step 2: Run all tests — verify nothing broke**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```
git add lemon_agent/pipeline.py
git commit -m "feat(pipeline): update run_full_pipeline — purpose param, universal FMs, batch rating"
```

---

## Task 7: End-to-end smoke test

**Files:**
- Read: `lemon_agent/run_pipeline.py`

- [ ] **Step 1: Run the existing sample script**

```
cd C:\Users\rohan\Downloads\ML\AnswerClustering
python -m lemon_agent.run_pipeline
```

Expected: the script completes, prints JSON output where each idea has a `ratings` list (not `evaluations`), `failure_mode_vector` is populated, and `is_lemon` is set. Some ideas should be flagged as lemons.

- [ ] **Step 2: Verify output shape**

Confirm the JSON output matches this structure for each idea:

```json
{
  "idea": "...",
  "ratings": [
    {"risk": "high|medium|low", "rationale": "..."},
    ...
  ],
  "failure_mode_vector": [1.0, 0.0, -1.0, ...],
  "is_lemon": true
}
```

- [ ] **Step 3: Commit**

```
git add lemon_agent/lemon_agent_output.json
git commit -m "test: end-to-end smoke test output for redesigned lemon agent"
```

---

## Self-Review

**Spec coverage:**
- ✅ Stage 2 universal FMs (no idea text): Task 4
- ✅ Stage 3 batch rating with rationale as CoT: Task 5
- ✅ `FailureModeRating` model: Task 1
- ✅ `IdeaEvaluation.ratings` replaces `evaluations`: Task 1
- ✅ `CriterionEvaluation` / `FailureModeResult` removed: Task 1
- ✅ `build_failure_modes_prompt` redesigned: Task 2
- ✅ `build_batch_rating_prompt` added: Task 2
- ✅ `build_batch_evaluation_prompt` removed: Task 2
- ✅ `purpose` optional string parameter threaded through all stages: Tasks 2, 4, 5, 6
- ✅ `rating_levels` validation (2 or 3 only, ValueError otherwise): Task 3
- ✅ Prompt enforces correct risk vocabulary per `rating_levels`: Task 2
- ✅ `_build_failure_mode_vector` updated to `List[FailureModeRating]`: Task 3
- ✅ `assign_lemon_labels` updated to use `.ratings`: Task 3
- ✅ `run_full_pipeline` new stage flow + `purpose` param: Task 6
- ✅ `evaluate_batch` removed: Task 5
- ✅ `_rating_to_score` removed: Task 4

**Type consistency check:**
- `_build_failure_mode_vector(ratings: List[FailureModeRating], ...)` used in Task 3, called in Task 3's `assign_lemon_labels` — consistent.
- `rate_ideas_batch` returns `List[IdeaEvaluation]` with `.ratings: List[FailureModeRating]` — consumed by `assign_lemon_labels` in Task 6 — consistent.
- `generate_failure_modes` returns `List[FailureMode]` — consumed by `rate_ideas_batch` in Task 6 — consistent.
- `build_batch_rating_prompt` defined in Task 2, imported in Task 3 (imports update), called in Task 5 — consistent.
