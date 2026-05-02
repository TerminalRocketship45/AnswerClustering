# Lemon Agent Redesign — 2026-05-01

## Goal

Improve the Lemon Agent so that failure mode vectors are directly comparable across all ideas, enabling valid Pareto dominance filtering and meaningful downstream clustering.

**Core invariant:** every idea must be rated against the exact same failure modes in the exact same order. Vector position `k` must mean the same thing for every idea.

---

## Pipeline Stages

### Stage 1 — Criteria Generation (unchanged)

- Input: random sample of ideas + optional `problem_statement`
- Output: `List[Criterion]` — k shared evaluation criteria
- Existing `generate_criteria` function is unchanged

### Stage 2 — Universal Failure Mode Generation (redesigned)

- Input: `List[Criterion]` + optional `purpose` (the GBSM context — the goal, barrier, or cause being solved)
- **No idea text is passed.** Failure modes are domain-level: what are the ways a solution in this domain could fail each criterion?
- Output: `List[FailureMode]` — a shared set of M failure modes where M = `k_criteria × failure_modes_per_criterion`
- The `risk` and `rationale` fields are removed from `FailureMode` — they are now idea-specific and live in Stage 3
- If `purpose` is provided, the prompt opens with a preamble mirroring `find_failure_modes_translated.py`: the purpose is treated as the GBSM tree context (e.g. "We are trying to achieve the goal X" or "We are trying to overcome the barrier X"), followed by the criteria list. If omitted, the prompt uses only the criteria and any available domain context.
- RAG/`gbsm_context` from `run_full_pipeline` is also forwarded here as supplementary context (same as today)

### Stage 3 — Batch Rating with Rationale (replaces evaluate_batch + old generate_failure_modes)

- Input: batch of ~10 ideas + shared `List[FailureMode]` + optional purpose/context
- The LLM sees the full batch simultaneously for cross-idea calibration
- For each idea × each shared failure mode, the LLM outputs:
  - `risk`: `"high"` / `"medium"` / `"low"`
  - `rationale`: 2–3 sentence explanation of why that risk rating applies to this specific idea (this is the chain-of-thought anchor, as in the original `find_failure_modes_translated.py`)
- Output: `List[IdeaEvaluation]`, each with `ratings: List[FailureModeRating]`
- Batches of ideas run sequentially; all use the same shared failure modes

---

## Vector Construction

After Stage 3, each `IdeaEvaluation` has a `ratings` list of length M (one per shared failure mode, in the same order for every idea).

```
risk_map = {"high": 1.0, "medium": 0.0, "low": -1.0}
failure_mode_vector[i] = risk_map[ratings[i].risk]
```

Lower values are better (lower risk). The Pareto dominance logic is unchanged: idea A is a lemon if any other idea B has equal-or-lower risk on all M dimensions and strictly lower risk on at least one.

---

## Optional Purpose Input

The user may optionally provide a `purpose` — a plain string describing the goal, barrier, or cause from the GBSM deliberation tree. Examples: `"achieve the goal of reducing urban carbon emissions"` or `"overcome the barrier of traffic congestion"`. The prompt builder injects it verbatim; no structured object is needed.

`purpose` is forwarded to Stage 2 (failure mode generation) and Stage 3 (batch rating) as framing context. It is added to `run_full_pipeline`'s signature as an optional `str` parameter.

---

## Data Model Changes (`models.py`)

### `FailureMode` — simplified

Remove: `solutionName`, `risk`, `rationale`

Keep: `type`, `criterionID`, `criterionName`, `name`, `description`

These are now universal domain objects, not tied to any specific idea.

### `FailureModeRating` — new

```python
@dataclass
class FailureModeRating:
    risk: Literal["high", "medium", "low"]
    rationale: str
```

One per (idea × shared failure mode), produced by Stage 3.

### `IdeaEvaluation` — updated

Replace `evaluations: List[CriterionEvaluation]` with `ratings: List[FailureModeRating]`.

`failure_mode_vector` and `is_lemon` are unchanged.

### Removed

`CriterionEvaluation` and `FailureModeResult` are removed entirely — they belonged to the old `evaluate_batch` design.

---

## Prompt Changes (`prompt_builder.py`)

### `build_failure_modes_prompt` — redesigned

- Parameters: `List[Criterion]`, optional `purpose`, optional `retrieved_context`
- No `ideas` parameter
- If `purpose` is provided, opens with a preamble derived from `explain_purpose` (e.g. "We are trying to achieve the goal X. These are the criteria a good solution would satisfy:")
- Output schema: list of failure modes without `risk` or `rationale` fields

### `build_batch_rating_prompt` — new

- Parameters: `List[str]` (ideas), `List[FailureMode]` (shared), optional `purpose`, optional `retrieved_context`, `rating_levels`
- Instructs the LLM to use the full batch for calibration but rate each idea independently
- Output schema per idea: `{idea, ratings: [{risk, rationale}]}` — one rating entry per failure mode, in the same order as the input failure modes list

### `build_batch_evaluation_prompt` — removed

---

## Pipeline Changes (`pipeline.py`)

### `generate_failure_modes` — redesigned

Signature: `generate_failure_modes(criteria, failure_modes_per_criterion, retrieved_context=None, purpose=None) -> List[FailureMode]`

No `ideas` parameter. Produces the shared failure mode set.

### `rate_ideas_batch` — new

Signature: `rate_ideas_batch(ideas, shared_failure_modes, retrieved_context=None, purpose=None) -> List[IdeaEvaluation]`

Calls `build_batch_rating_prompt`, parses response, populates `IdeaEvaluation.ratings`.

### `evaluate_batch` — removed

### `_build_failure_mode_vector` — updated

Signature changes from `(List[FailureMode], expected_length)` to `(List[FailureModeRating], expected_length)`. Reads `rating.risk` instead of `fm.risk`. The expected length calculation (`k_criteria × failure_modes_per_criterion`) is unchanged.

### `run_full_pipeline` — updated

New `purpose` parameter. New stage flow:

```
1. generate_criteria(sampled_ideas, k, problem_statement)
2. [optional] build RAG context
3. generate_failure_modes(criteria, n, context, purpose)   # shared, no ideas
4. for each batch of ideas:
       rate_ideas_batch(batch, shared_fms, context, purpose)
5. assign_lemon_labels(all_evaluations, criteria)
```

---

## Rating Levels (config-driven)

`config["rating_levels"]` controls how many risk buckets the LLM uses in Stage 3. Only two values are valid:

| `rating_levels` | Allowed risk values | Vector mapping |
|---|---|---|
| `2` | `"high"`, `"low"` | high → 1.0, low → -1.0 |
| `3` | `"high"`, `"medium"`, `"low"` | high → 1.0, medium → 0.0, low → -1.0 |

Any other value raises a `ValueError` at startup (in `run_full_pipeline` before any LLM calls). The `build_batch_rating_prompt` injects the valid values into the prompt so the LLM knows exactly what it is allowed to output (e.g. "Use only: high or low" for `rating_levels=2`, or "Use only: high, medium, or low" for `rating_levels=3`).

---

## What Does Not Change

- `generate_criteria` and `build_criteria_prompt`
- `assign_lemon_labels` and `_is_pareto_dominated`
- `config.py` structure (values are read, not restructured)
- `rag.py`
- `llm_client.py`
- The `Criterion` model
