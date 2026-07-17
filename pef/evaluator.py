"""Core evaluation orchestration: run prompt versions against test cases and score them."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import PromptConstraints
from .criteria import (
    CRITERIA,
    score_accuracy,
    score_clarity,
    score_consistency,
    score_instruction_following,
    score_relevance,
)


@dataclass
class PromptVersion:
    id: str
    text: str
    constraints: PromptConstraints = field(default_factory=PromptConstraints)


@dataclass
class EvalCase:
    id: str
    input_text: str
    key_facts: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    prompt_id: str
    case_id: str
    primary_output: str
    all_outputs: list[str]
    scores: dict[str, float]


@dataclass
class PromptVersionResult:
    prompt_id: str
    case_results: list[CaseResult]
    mean_scores: dict[str, float]
    composite: float


DEFAULT_WEIGHTS: dict[str, float] = {c: 1.0 for c in CRITERIA}


class PromptEvaluator:
    """Runs each prompt version against every test case and aggregates scores."""

    def __init__(self, backend, runs_per_case: int = 3, weights: dict[str, float] | None = None) -> None:
        if runs_per_case < 1:
            raise ValueError("runs_per_case must be >= 1")
        self.backend = backend
        self.runs_per_case = runs_per_case
        self.weights = weights or DEFAULT_WEIGHTS

    def _composite(self, scores: dict[str, float]) -> float:
        total_weight = sum(self.weights.get(c, 0.0) for c in CRITERIA)
        if total_weight == 0:
            return 0.0
        return sum(scores[c] * self.weights.get(c, 0.0) for c in CRITERIA) / total_weight

    def _evaluate_case(self, prompt: PromptVersion, case: EvalCase) -> CaseResult:
        outputs = [
            self.backend.generate(
                prompt.text,
                prompt.constraints,
                case.input_text,
                run_index=i,
                prompt_id=prompt.id,
                input_id=case.id,
            )
            for i in range(self.runs_per_case)
        ]
        primary = outputs[0]

        scores = {
            "accuracy": score_accuracy(primary, case.key_facts),
            "relevance": score_relevance(primary, case.input_text),
            "clarity": score_clarity(primary),
            "consistency": score_consistency(outputs),
            "instruction_following": score_instruction_following(primary, prompt.constraints),
        }
        return CaseResult(
            prompt_id=prompt.id,
            case_id=case.id,
            primary_output=primary,
            all_outputs=outputs,
            scores=scores,
        )

    def evaluate(
        self, prompt_versions: list[PromptVersion], test_cases: list[EvalCase]
    ) -> list[PromptVersionResult]:
        results: list[PromptVersionResult] = []
        for prompt in prompt_versions:
            case_results = [self._evaluate_case(prompt, case) for case in test_cases]

            mean_scores = {
                c: sum(cr.scores[c] for cr in case_results) / len(case_results) for c in CRITERIA
            }
            composite = self._composite(mean_scores)

            results.append(
                PromptVersionResult(
                    prompt_id=prompt.id,
                    case_results=case_results,
                    mean_scores=mean_scores,
                    composite=composite,
                )
            )
        return results
