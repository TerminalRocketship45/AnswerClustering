from __future__ import annotations

import json
import logging
import math
import random
from typing import Any, Dict, List, Optional

from .config import config
from .llm_client import generate_json
from .models import Criterion, Feature, FeatureResult, CriterionEvaluation, IdeaEvaluation
from .prompt_builder import (
    SYSTEM_PROMPT,
    build_batch_evaluation_prompt,
    build_criteria_prompt,
    build_feature_prompt,
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


def generate_features(
    criteria: List[Criterion],
    features_per_criterion: int,
) -> List[Feature]:
    prompt = build_feature_prompt(features_per_criterion, criteria)
    raw_features = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )
    features: List[Feature] = []
    for item in raw_features:
        features.append(
            Feature(
                criterion_id=item.get("criterionID", ""),
                criterion_name=item.get("criterionName", ""),
                question=item.get("question", ""),
            )
        )
    return features


def _group_features_by_criterion(features: List[Feature]) -> Dict[str, List[Feature]]:
    grouped: Dict[str, List[Feature]] = {}
    for feature in features:
        grouped.setdefault(feature.criterion_id, []).append(feature)
    return grouped


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
    features: List[Feature],
    rating_levels: int,
    retrieved_context: Optional[str] = None,
) -> List[IdeaEvaluation]:
    prompt = build_batch_evaluation_prompt(ideas, features, rating_levels, retrieved_context)
    raw_evaluations = generate_json(
        prompt,
        model=config["model"],
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=config["max_llm_tokens"],
    )

    grouped_features = _group_features_by_criterion(features)
    results: List[IdeaEvaluation] = []
    for idea_result in raw_evaluations:
        idea_text = idea_result.get("idea", "")
        evaluations: List[CriterionEvaluation] = []
        for ev in idea_result.get("evaluations", []):
            feature_results: List[FeatureResult] = []
            for feature_item in ev.get("features", []):
                feature_results.append(
                    FeatureResult(
                        feature=feature_item.get("feature", ""),
                        rating=feature_item.get("rating", "no"),
                        reasoning=feature_item.get("reasoning", ""),
                    )
                )
            evaluations.append(
                CriterionEvaluation(
                    criterionID=ev.get("criterionID", ""),
                    criterionName=ev.get("criterionName", ""),
                    features=feature_results,
                )
            )
        results.append(IdeaEvaluation(idea=idea_text, evaluations=evaluations))
    return results


def _build_feature_vector(evaluation: IdeaEvaluation, features: List[Feature]) -> List[float]:
    question_to_rating: Dict[str, str] = {}
    for criterion_eval in evaluation.evaluations:
        for feature_item in criterion_eval.features:
            question_to_rating[feature_item.feature] = feature_item.rating

    vector: List[float] = []
    for feature in features:
        rating = question_to_rating.get(feature.question, "no")
        vector.append(_rating_to_score(rating, config["rating_levels"]))
    return vector


def _is_pareto_dominated(target: List[float], candidate: List[float]) -> bool:
    better_or_equal = all(c <= t + 1e-9 for t, c in zip(target, candidate))
    strictly_better = any(c < t - 1e-9 for t, c in zip(target, candidate))
    return better_or_equal and strictly_better


def assign_lemon_labels(
    evaluations: List[IdeaEvaluation],
    features: List[Feature],
) -> List[IdeaEvaluation]:
    vectors = [_build_feature_vector(eval_item, features) for eval_item in evaluations]
    for idea_eval, vector in zip(evaluations, vectors):
        idea_eval.feature_vector = vector

    for idx, idea_eval in enumerate(evaluations):
        computed = False
        for jdx, other_vector in enumerate(vectors):
            if idx == jdx:
                continue
            if _is_pareto_dominated(vector, other_vector):
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
) -> List[Dict[str, Any]]:
    sampled_ideas = sample_ideas(ideas, config["criteria_sample_size"])
    criteria = generate_criteria(sampled_ideas, config["k_criteria"], problem_statement)
    features = generate_features(criteria, config["features_per_criterion"])

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

    batches: List[List[str]] = []
    for i in range(0, len(ideas), config["batch_size"]):
        batches.append(ideas[i : i + config["batch_size"]])

    all_evaluations: List[IdeaEvaluation] = []
    for batch in batches:
        batch_context = retrieved_context if config["rag_enabled"] else None
        batch_results = evaluate_batch(batch, features, config["rating_levels"], batch_context)
        all_evaluations.extend(batch_results)

    final_evaluations = assign_lemon_labels(all_evaluations, features)
    return [item.to_dict() for item in final_evaluations]


def save_output(output: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
