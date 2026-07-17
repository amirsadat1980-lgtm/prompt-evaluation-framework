from pef.backends import MockBackend
from pef.evaluator import PromptEvaluator
from pef.loaders import load_prompt_versions, load_test_cases
from pef.report import to_json, to_markdown
from tests.test_evaluator import PROMPTS_MANIFEST, TEST_CASES_PATH


def _run():
    versions = load_prompt_versions(PROMPTS_MANIFEST)
    cases = load_test_cases(TEST_CASES_PATH)
    evaluator = PromptEvaluator(backend=MockBackend(), runs_per_case=2)
    return evaluator.evaluate(versions, cases)


def test_to_markdown_lists_every_prompt_and_is_ranked_by_composite():
    results = _run()
    markdown = to_markdown(results)

    for r in results:
        assert r.prompt_id in markdown

    ranked = sorted(results, key=lambda r: r.composite, reverse=True)
    positions = [markdown.index(r.prompt_id) for r in ranked]
    assert positions == sorted(positions)


def test_to_json_round_trips_scores():
    results = _run()
    payload = to_json(results)

    assert set(payload["criteria"]) == {
        "accuracy",
        "relevance",
        "clarity",
        "consistency",
        "instruction_following",
    }
    assert len(payload["results"]) == len(results)
    for entry in payload["results"]:
        assert 0.0 <= entry["composite"] <= 1.0
        assert len(entry["cases"]) == 3
