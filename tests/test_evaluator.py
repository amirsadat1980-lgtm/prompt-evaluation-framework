from pathlib import Path

import pytest

from pef.backends import MockBackend, PromptConstraints
from pef.criteria import CRITERIA
from pef.evaluator import PromptEvaluator, PromptVersion, EvalCase
from pef.loaders import load_prompt_versions, load_test_cases

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_MANIFEST = ROOT / "examples" / "prompts" / "prompts.json"
TEST_CASES_PATH = ROOT / "examples" / "test_cases.json"


def test_loaders_read_the_bundled_examples():
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)

    assert [v.id for v in versions] == ["v1_naive", "v2_role_primed", "v3_few_shot_strict"]
    assert len(cases) == 3
    assert all(case.key_facts for case in cases)


def test_evaluator_rejects_invalid_runs_per_case():
    with pytest.raises(ValueError):
        PromptEvaluator(backend=MockBackend(), runs_per_case=0)


def test_evaluate_produces_a_result_per_prompt_with_all_criteria_scored():
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)

    evaluator = PromptEvaluator(backend=MockBackend(), runs_per_case=3)
    results = evaluator.evaluate(versions, cases)

    assert len(results) == len(versions)
    for result in results:
        assert set(result.mean_scores.keys()) == set(CRITERIA)
        assert 0.0 <= result.composite <= 1.0
        for score in result.mean_scores.values():
            assert 0.0 <= score <= 1.0
        assert len(result.case_results) == len(cases)


def test_stricter_prompts_are_at_least_as_clear_as_the_naive_prompt():
    """v2/v3 explicitly ask for plain language and a hard length limit; the
    naive v1 prompt asks for neither, so it should never beat them on clarity."""
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)

    evaluator = PromptEvaluator(backend=MockBackend(), runs_per_case=3)
    results = {r.prompt_id: r for r in evaluator.evaluate(versions, cases)}

    assert results["v2_role_primed"].mean_scores["clarity"] >= results["v1_naive"].mean_scores["clarity"]
    assert results["v3_few_shot_strict"].mean_scores["clarity"] >= results["v1_naive"].mean_scores["clarity"]


def test_stricter_prompts_are_at_least_as_consistent_as_the_naive_prompt():
    """Explicit constraints should reduce run-to-run variance versus an
    underspecified prompt."""
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)

    evaluator = PromptEvaluator(backend=MockBackend(), runs_per_case=5)
    results = {r.prompt_id: r for r in evaluator.evaluate(versions, cases)}

    assert (
        results["v3_few_shot_strict"].mean_scores["consistency"]
        >= results["v1_naive"].mean_scores["consistency"]
    )


def test_instruction_following_is_perfect_when_backend_honors_constraints():
    """MockBackend always satisfies its own primary-run constraints by
    construction, so instruction-following should be a perfect score."""
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)

    evaluator = PromptEvaluator(backend=MockBackend(), runs_per_case=1)
    results = evaluator.evaluate(versions, cases)

    for result in results:
        assert result.mean_scores["instruction_following"] == 1.0


def test_custom_weights_change_the_composite_ranking():
    versions = [
        PromptVersion(id="a", text="x", constraints=PromptConstraints(max_sentences=1)),
    ]
    cases = [EvalCase(id="c1", input_text="One sentence here. Another one too.", key_facts=[])]

    evaluator_equal = PromptEvaluator(backend=MockBackend(), runs_per_case=1)
    equal_result = evaluator_equal.evaluate(versions, cases)[0]

    weights = {c: 0.0 for c in CRITERIA}
    weights["clarity"] = 1.0
    evaluator_weighted = PromptEvaluator(backend=MockBackend(), runs_per_case=1, weights=weights)
    weighted_result = evaluator_weighted.evaluate(versions, cases)[0]

    assert weighted_result.composite == pytest.approx(weighted_result.mean_scores["clarity"])
    assert equal_result.composite == pytest.approx(
        sum(equal_result.mean_scores.values()) / len(equal_result.mean_scores)
    )
