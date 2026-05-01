from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, List, Optional

Rating2 = Literal["yes", "no"]
Rating3 = Literal["high", "medium", "low"]
RatingValue = Literal["yes", "no", "high", "medium", "low"]


@dataclass
class Criterion:
    eid: str
    name: str
    description: str


@dataclass
class Feature:
    criterion_id: str
    criterion_name: str
    question: str


@dataclass
class FeatureResult:
    feature: str
    rating: RatingValue
    reasoning: str


@dataclass
class CriterionEvaluation:
    criterionID: str
    criterionName: str
    features: List[FeatureResult] = field(default_factory=list)


@dataclass
class IdeaEvaluation:
    idea: str
    evaluations: List[CriterionEvaluation] = field(default_factory=list)
    feature_vector: List[float] = field(default_factory=list)
    is_lemon: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea,
            "evaluations": [
                {
                    "criterionID": ev.criterionID,
                    "criterionName": ev.criterionName,
                    "features": [
                        {
                            "feature": fr.feature,
                            "rating": fr.rating,
                            "reasoning": fr.reasoning,
                        }
                        for fr in ev.features
                    ],
                }
                for ev in self.evaluations
            ],
            "feature_vector": self.feature_vector,
            "is_lemon": self.is_lemon,
        }
