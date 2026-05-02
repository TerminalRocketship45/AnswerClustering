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
class FailureMode:
    type: str
    solutionName: str
    criterionID: str
    criterionName: str
    name: str
    description: str
    risk: Literal["high", "medium", "low"]
    rationale: str


@dataclass
class FailureModeResult:
    failure_mode: str
    rating: RatingValue
    reasoning: str


@dataclass
class CriterionEvaluation:
    criterionID: str
    criterionName: str
    failure_modes: List[FailureModeResult] = field(default_factory=list)


@dataclass
class IdeaEvaluation:
    idea: str
    evaluations: List[CriterionEvaluation] = field(default_factory=list)
    failure_modes: List[FailureMode] = field(default_factory=list)
    failure_mode_vector: List[float] = field(default_factory=list)
    is_lemon: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea,
            "evaluations": [
                {
                    "criterionID": ev.criterionID,
                    "criterionName": ev.criterionName,
                    "failureModes": [
                        {
                            "failureMode": fr.failure_mode,
                            "rating": fr.rating,
                            "reasoning": fr.reasoning,
                        }
                        for fr in ev.failure_modes
                    ],
                }
                for ev in self.evaluations
            ],
            "failure_modes": [
                {
                    "type": fm.type,
                    "solutionName": fm.solutionName,
                    "criterionID": fm.criterionID,
                    "criterionName": fm.criterionName,
                    "name": fm.name,
                    "description": fm.description,
                    "risk": fm.risk,
                    "rationale": fm.rationale,
                }
                for fm in self.failure_modes
            ],
            "failure_mode_vector": self.failure_mode_vector,
            "is_lemon": self.is_lemon,
        }
