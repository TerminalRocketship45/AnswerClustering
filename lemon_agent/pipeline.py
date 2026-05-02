from __future__ import annotations

import json
import logging
import math
import random
from typing import Any, Dict, List, Optional

from .config import config
from .llm_client import generate_json
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
from .rag import build_document_store, retrieve_context, search_fallback


def sample_ideas(ideas: List[str], sample_size: int) -> List[str]:
    if len(ideas) <= sample_size:
        return list(ideas)
    return random.sample(ideas, sample_size)


def generate_criteria(
    ideas: List[str],
    k_criteria: int,
    problem_statement: Optional[str] = None,
) -> List[Criterion]:
    prompt = build_criteria_prompt(ideas, k_criteria, problem_statement)
    raw_criteria = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    criteria: List[Criterion] = []
    for index, item in enumerate(raw_criteria, start=1):
        criterion_id = item.get("id") or f"C{index}"
        criteria.append(
            Criterion(
                eid=criterion_id,
                name=item.get("name", f"Criterion {index}"),
                description=item.get("description", ""),
            )
        )
    return criteria


def generate_failure_modes(
    ideas: List[str],
    criteria: List[Criterion],
    failure_modes_per_criterion: int,
    retrieved_context: Optional[str] = None,
    hint: Optional[str] = None,
) -> List[List[FailureMode]]:
    prompt = build_failure_modes_prompt(
        ideas,
        criteria,
        failure_modes_per_criterion,
        retrieved_context,
        hint,
        style=config.get("failure_mode_style", "statement"),
    )
    raw_failure_modes = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    grouped: Dict[str, List[FailureMode]] = {idea: [] for idea in ideas}
    for item in raw_failure_modes:
        solution_name = item.get("solutionName", "")
        if solution_name not in grouped:
            grouped[solution_name] = []
        grouped[solution_name].append(
            FailureMode(
                type=item.get("type", "failure"),
                solutionName=solution_name,
                criterionID=item.get("criterionID", ""),
                criterionName=item.get("criterionName", ""),
                name=item.get("name", ""),
                description=item.get("description", ""),
                risk=str(item.get("risk", "low")).strip().lower(),
                rationale=item.get("rationale", ""),
            )
        )

    return [grouped.get(idea, []) for idea in ideas]


def _rating_to_score(rating: str, rating_levels: int) -> float:
    normalized = rating.strip().lower()
    if rating_levels == 2:
        if normalized == "yes":
            return 1.0
        if normalized == "no":
            return 0.0
    else:
        if normalized == "high":
            return 1.0
        if normalized == "medium":
            return 0.5
        if normalized == "low":
            return 0.0
    return 0.0


def evaluate_batch(
    ideas: List[str],
    failure_modes: List[FailureMode],
    rating_levels: int,
    retrieved_context: Optional[str] = None,
) -> List[IdeaEvaluation]:
    prompt = build_batch_evaluation_prompt(
        ideas,
        [
            {
                "criterionID": fm.criterionID,
                "criterionName": fm.criterionName,
                "name": fm.name,
                "description": fm.description,
            }
            for fm in failure_modes
        ],
        rating_levels,
        retrieved_context,
    )
    raw_evaluations = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    results: List[IdeaEvaluation] = []
    for idea_result in raw_evaluations:
        idea_text = idea_result.get("idea", "")
        evaluations: List[CriterionEvaluation] = []
        for ev in idea_result.get("evaluations", []):
            failure_mode_results: List[FailureModeResult] = []
            for failure_item in ev.get("failureModes", []):
                failure_mode_results.append(
                    FailureModeResult(
                        failure_mode=failure_item.get("failureMode", ""),
                        rating=failure_item.get("rating", "no"),
                        reasoning=failure_item.get("reasoning", ""),
                    )
                )
            evaluations.append(
                CriterionEvaluation(
                    criterionID=ev.get("criterionID", ""),
                    criterionName=ev.get("criterionName", ""),
                    failure_modes=failure_mode_results,
                )
            )
        results.append(IdeaEvaluation(idea=idea_text, evaluations=evaluations))
    return results


def _build_failure_mode_vector(
    failure_modes: List[FailureMode],
    expected_length: int,
) -> List[float]:
    risk_map = {
        "high": 1.0,
        "medium": 0.0,
        "low": -1.0,
    }
    vector = [risk_map.get(fm.risk, 0.0) for fm in failure_modes]
    if len(vector) < expected_length:
        vector.extend([0.0] * (expected_length - len(vector)))
    return vector[:expected_length]


def _is_pareto_dominated(target: List[float], candidate: List[float]) -> bool:
    better_or_equal = all(c <= t + 1e-9 for t, c in zip(target, candidate))
    strictly_better = any(c < t - 1e-9 for t, c in zip(target, candidate))
    return better_or_equal and strictly_better


def assign_lemon_labels(
    evaluations: List[IdeaEvaluation],
    criteria: List[Criterion],
) -> List[IdeaEvaluation]:
    expected_len = len(criteria) * config["failure_modes_per_criterion"]
    vectors = [
        _build_failure_mode_vector(eval_item.failure_modes, expected_len)
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


def run_full_pipeline(
    ideas: List[str],
    problem_statement: Optional[str] = None,
    documents: Optional[List[str]] = None,
    gbsm_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sampled_ideas = sample_ideas(ideas, config["criteria_sample_size"])
    criteria = generate_criteria(sampled_ideas, config["k_criteria"], problem_statement)

    document_store = []
    retrieved_context = ""
    if config["rag_enabled"]:
        documents = documents or []
        if documents:
            document_store = build_document_store(documents, model_name=config["embeddings_model"])
            retrieved_context = retrieve_context(
                problem_statement or "", document_store, model_name=config["embeddings_model"], top_k=config["rag_top_k"]
            )
        else:
            retrieved_context = search_fallback(problem_statement or "", model_name=config["model"])

    if gbsm_context and retrieved_context:
        failure_mode_context = f"{gbsm_context}\n\n{retrieved_context}"
    else:
        failure_mode_context = gbsm_context or retrieved_context or None

    batches: List[List[str]] = []
    for i in range(0, len(ideas), config["batch_size"]):
        batches.append(ideas[i : i + config["batch_size"]])

    all_failure_modes: List[List[FailureMode]] = []
    for batch in batches:
        batch_failure_modes = generate_failure_modes(
            batch,
            criteria,
            config["failure_modes_per_criterion"],
            failure_mode_context,
        )
        all_failure_modes.extend(batch_failure_modes)

    all_evaluations: List[IdeaEvaluation] = []
    for batch, batch_failure_modes in zip(batches, all_failure_modes):
        batch_context = retrieved_context if config["rag_enabled"] else None
        batch_results = evaluate_batch(batch, batch_failure_modes, config["rating_levels"], batch_context)
        for evaluation, failure_modes in zip(batch_results, batch_failure_modes):
            evaluation.failure_modes = failure_modes
        all_evaluations.extend(batch_results)

    final_evaluations = assign_lemon_labels(all_evaluations, criteria)
    return [item.to_dict() for item in final_evaluations]


def save_output(output: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
