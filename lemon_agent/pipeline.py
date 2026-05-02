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
    FailureModeRating,
    IdeaEvaluation,
)
from .prompt_builder import (
    SYSTEM_PROMPT,
    build_batch_rating_prompt,
    build_criteria_prompt,
    build_failure_modes_prompt,
)
from .rag import build_document_store, retrieve_context, search_fallback


def sample_ideas(ideas: List[str], sample_size: int) -> List[str]:
    if len(ideas) <= sample_size:
        return list(ideas)
    return random.sample(ideas, sample_size)


def _dry_run_risk() -> str:
    return "yes" if config["rating_levels"] == 2 else "low"


def generate_criteria(
    ideas: List[str],
    k_criteria: int,
    problem_statement: Optional[str] = None,
    criteria_list: Optional[List[str]] = None,
    dry_run: bool = False,
) -> tuple[List[Criterion], dict[str, Any]]:
    if criteria_list:
        # For multi-agent, choose from list
        prompt = build_criteria_prompt(ideas, k_criteria, problem_statement, criteria_list)
        if dry_run:
            criteria = [
                Criterion(
                    eid=f"C{index}",
                    name=name,
                    description=f"Dry-run placeholder for criterion '{name}'.",
                )
                for index, name in enumerate(criteria_list[:k_criteria], start=1)
            ]
            raw_response = "DRY_RUN: simulated criteria generation"
            raw_criteria = [
                {
                    "id": criterion.eid,
                    "name": criterion.name,
                    "description": criterion.description,
                }
                for criterion in criteria
            ]
        else:
            raw_response, raw_criteria = generate_json(
                prompt,
                model=config["model"],
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=config["max_llm_tokens"],
                return_raw=True,
            )
            criteria = []
            for index, item in enumerate(raw_criteria, start=1):
                name = item.get("name", f"Criterion {index}")
                criteria.append(
                    Criterion(
                        eid=f"C{index}",
                        name=name,
                        description=item.get("description", ""),
                    )
                )
    else:
        # Single-agent, generate
        prompt = build_criteria_prompt(ideas, k_criteria, problem_statement)
        if dry_run:
            criteria = [
                Criterion(
                    eid=f"C{index}",
                    name=f"Criterion {index}",
                    description=f"Dry-run placeholder criterion {index}.",
                )
                for index in range(1, k_criteria + 1)
            ]
            raw_response = "DRY_RUN: simulated criteria generation"
            raw_criteria = [
                {
                    "id": criterion.eid,
                    "name": criterion.name,
                    "description": criterion.description,
                }
                for criterion in criteria
            ]
        else:
            raw_response, raw_criteria = generate_json(
                prompt,
                model=config["model"],
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=config["max_llm_tokens"],
                return_raw=True,
            )

            criteria = []
            for index, item in enumerate(raw_criteria, start=1):
                criterion_id = item.get("id") or f"C{index}"
                criteria.append(
                    Criterion(
                        eid=criterion_id,
                        name=item.get("name", f"Criterion {index}"),
                        description=item.get("description", ""),
                    )
                )
    return criteria, {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": raw_criteria,
    }


def generate_failure_modes(
    criteria: List[Criterion],
    failure_modes_per_criterion: int,
    retrieved_context: Optional[str] = None,
    purpose: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[List[FailureMode], dict[str, Any]]:
    prompt = build_failure_modes_prompt(
        criteria,
        failure_modes_per_criterion,
        retrieved_context,
        purpose,
        style=config.get("failure_mode_style", "statement"),
    )
    if dry_run:
        failure_modes: List[FailureMode] = []
        for criterion in criteria:
            for index in range(1, failure_modes_per_criterion + 1):
                failure_modes.append(
                    FailureMode(
                        type="failure",
                        criterionID=criterion.eid,
                        criterionName=criterion.name,
                        name=f"Dry-run failure mode {index} for {criterion.name}",
                        description="Dry-run placeholder failure mode.",
                    )
                )
        raw_response = "DRY_RUN: simulated failure mode generation"
        raw_failure_modes = [
            {
                "type": fm.type,
                "criterionID": fm.criterionID,
                "criterionName": fm.criterionName,
                "name": fm.name,
                "description": fm.description,
            }
            for fm in failure_modes
        ]
    else:
        raw_response, raw_failure_modes = generate_json(
            prompt,
            model=config["model"],
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=config["max_llm_tokens"],
            return_raw=True,
        )

        failure_modes = []
        for item in raw_failure_modes:
            failure_modes.append(
                FailureMode(
                    type=item.get("type", "failure"),
                    criterionID=item.get("criterionID", ""),
                    criterionName=item.get("criterionName", ""),
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                )
            )
    return failure_modes, {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": raw_failure_modes,
    }


def rate_ideas_batch(
    ideas: List[str],
    failure_modes: List[FailureMode],
    retrieved_context: Optional[str] = None,
    purpose: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[List[IdeaEvaluation], dict[str, Any]]:
    prompt = build_batch_rating_prompt(
        ideas,
        failure_modes,
        purpose,
        retrieved_context,
        config["rating_levels"],
        config["rationale"],
    )
    if dry_run:
        results: List[IdeaEvaluation] = []
        raw_evaluations: List[dict[str, Any]] = []
        for idea in ideas:
            ratings: List[FailureModeRating] = [
                FailureModeRating(risk=_dry_run_risk(), rationale="Dry-run placeholder rating.")
                for _ in failure_modes
            ]
            results.append(IdeaEvaluation(idea=idea, ratings=ratings))
            raw_evaluations.append(
                {
                    "idea": idea,
                    "ratings": [
                        {"risk": rating.risk, "rationale": rating.rationale}
                        for rating in ratings
                    ],
                }
            )
        raw_response = "DRY_RUN: simulated batch ratings"
    else:
        raw_response, raw_evaluations = generate_json(
            prompt,
            model=config["model"],
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=config["max_llm_tokens"],
            return_raw=True,
        )

        results = []
        for idea_result in raw_evaluations:
            idea_text = idea_result.get("idea", "")
            ratings_data = idea_result.get("ratings", [])
            ratings: List[FailureModeRating] = []
            for rating_item in ratings_data:
                if isinstance(rating_item, str):
                    # No rationale
                    ratings.append(FailureModeRating(risk=rating_item.lower()))
                else:
                    # With rationale
                    ratings.append(FailureModeRating(
                        risk=rating_item.get("risk", "low").lower(),
                        rationale=rating_item.get("rationale")
                    ))
            results.append(IdeaEvaluation(idea=idea_text, ratings=ratings))
    return results, {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": raw_evaluations,
    }


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
    ratings: List[FailureModeRating],
    expected_length: int,
) -> List[float]:
    risk_map = {
        "high": 1.0,
        "medium": 0.0,
        "low": -1.0,
    }
    vector = [risk_map.get(rating.risk, 0.0) for rating in ratings]
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
    shared_failure_modes: List[FailureMode],
) -> List[IdeaEvaluation]:
    expected_len = len(shared_failure_modes)
    failure_vectors = [
        _build_failure_mode_vector(eval_item.ratings, expected_len)
        for eval_item in evaluations
    ]

    fms_per_criterion = config["failure_modes_per_criterion"]
    for eval_item, f_vector in zip(evaluations, failure_vectors):
        if config["plot_mode"] == "criteria":
            criteria_vector = []
            for i in range(len(criteria)):
                start = i * fms_per_criterion
                end = start + fms_per_criterion
                criterion_ratings = eval_item.ratings[start:end]
                has_high = any(r.risk == "high" for r in criterion_ratings)
                criteria_vector.append(0.0 if has_high else 1.0)
            eval_item.failure_mode_vector = criteria_vector
        else:
            eval_item.failure_mode_vector = f_vector

    vectors = [eval_item.failure_mode_vector for eval_item in evaluations]

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
    purpose: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    # Validation
    if config["rating_levels"] not in [2, 3]:
        raise ValueError("rating_levels must be 2 or 3")
    if config["architecture"] == "multi":
        if not config["criteria_list"] or not config["fewshot_samples_path"]:
            raise ValueError("For multi-agent, criteria_list and fewshot_samples_path must be provided")
        # Load few-shot samples
        import json
        with open(config["fewshot_samples_path"], "r") as f:
            fewshot_data = json.load(f)
        # Assume fewshot_data is dict[criterion_name: list[dict]]

    sampled_ideas = sample_ideas(ideas, config["criteria_sample_size"])
    criteria_list = config["criteria_list"] if config["architecture"] == "multi" else None
    criteria, criteria_output = generate_criteria(
        sampled_ideas,
        config["k_criteria"],
        problem_statement,
        criteria_list,
        dry_run=dry_run,
    )

    document_store = []
    retrieved_context = ""
    if config["search_enabled"]:
        if dry_run:
            retrieved_context = "DRY_RUN: search and retrieval skipped in dry-run mode."
        else:
            documents = documents or []
            if documents:
                document_store = build_document_store(documents, model_name=config["embeddings_model"])
                retrieved_context = retrieve_context(
                    problem_statement or "", document_store, model_name=config["embeddings_model"], top_k=config["rag_top_k"]
                )
            else:
                retrieved_context = search_fallback(problem_statement or "", model_name=config["model"])

    if gbsm_context and retrieved_context:
        context = f"{gbsm_context}\n\n{retrieved_context}"
    else:
        context = gbsm_context or retrieved_context or None

    # Generate shared failure modes
    shared_failure_modes, failure_modes_output = generate_failure_modes(
        criteria,
        config["failure_modes_per_criterion"],
        context,
        purpose,
        dry_run=dry_run,
    )

    # Rate all ideas in batches
    all_evaluations: List[IdeaEvaluation] = []
    rating_outputs: List[dict[str, Any]] = []
    for i in range(0, len(ideas), config["batch_size"]):
        batch = ideas[i : i + config["batch_size"]]
        batch_evaluations, batch_output = rate_ideas_batch(
            batch,
            shared_failure_modes,
            context,
            purpose,
            dry_run=dry_run,
        )
        all_evaluations.extend(batch_evaluations)
        rating_outputs.append(batch_output)

    final_evaluations = assign_lemon_labels(all_evaluations, criteria, shared_failure_modes)

    feature_vector_order = [
        f"{fm.criterionID}: {fm.criterionName} — {fm.name}"
        for fm in shared_failure_modes
    ]

    idea_results = []
    for eval_item in final_evaluations:
        ratings = []
        for rating_item, failure_mode in zip(eval_item.ratings, shared_failure_modes):
            ratings.append(
                {
                    "criterionID": failure_mode.criterionID,
                    "criterionName": failure_mode.criterionName,
                    "failureModeName": failure_mode.name,
                    "failureModeDescription": failure_mode.description,
                    "risk": rating_item.risk,
                    "rationale": rating_item.rationale,
                }
            )
        idea_results.append(
            {
                "idea": eval_item.idea,
                "ratings": ratings,
                "failure_mode_vector": eval_item.failure_mode_vector,
                "feature_vector_keys": feature_vector_order,
                "is_lemon": eval_item.is_lemon,
            }
        )

    return {
        "criteria": [
            {
                "id": criterion.eid,
                "name": criterion.name,
                "description": criterion.description,
            }
            for criterion in criteria
        ],
        "failure_modes": [
            {
                "type": fm.type,
                "criterionID": fm.criterionID,
                "criterionName": fm.criterionName,
                "name": fm.name,
                "description": fm.description,
            }
            for fm in shared_failure_modes
        ],
        "feature_vector_order": feature_vector_order,
        "ideas": idea_results,
        "model_outputs": {
            "criteria_generation": criteria_output,
            "failure_mode_generation": failure_modes_output,
            "rating_batches": rating_outputs,
        },
    }


def save_output(output: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
