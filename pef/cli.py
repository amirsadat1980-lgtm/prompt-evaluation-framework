"""Command-line entry point.

    python -m pef.cli evaluate
    python -m pef.cli evaluate --runs-per-case 5 --output-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import MockBackend
from .evaluator import PromptEvaluator
from .loaders import load_prompt_versions, load_test_cases
from .report import to_json, to_markdown

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "examples" / "prompts" / "prompts.json"
DEFAULT_TEST_CASES = ROOT / "examples" / "test_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pef", description="Prompt Evaluation Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="Run all prompt versions against all test cases")
    evaluate.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Path to prompts.json manifest")
    evaluate.add_argument("--test-cases", type=Path, default=DEFAULT_TEST_CASES, help="Path to test_cases.json")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write reports")
    evaluate.add_argument("--runs-per-case", type=int, default=3, help="Repeated generations per test case (for consistency scoring)")
    evaluate.add_argument("--backend", choices=["mock", "openai"], default="mock")
    evaluate.add_argument("--quiet", action="store_true", help="Don't print the markdown table to stdout")

    return parser


def _build_backend(name: str):
    if name == "mock":
        return MockBackend()
    if name == "openai":
        from .backends import OpenAIBackend

        return OpenAIBackend()
    raise ValueError(f"Unknown backend: {name}")


def run_evaluate(args: argparse.Namespace) -> int:
    prompt_versions = load_prompt_versions(args.prompts)
    test_cases = load_test_cases(args.test_cases)
    backend = _build_backend(args.backend)

    evaluator = PromptEvaluator(backend=backend, runs_per_case=args.runs_per_case)
    results = evaluator.evaluate(prompt_versions, test_cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(results)
    (args.output_dir / "sample_results.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "sample_results.json").write_text(
        json.dumps(to_json(results), indent=2), encoding="utf-8"
    )

    if not args.quiet:
        print(markdown)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return run_evaluate(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
