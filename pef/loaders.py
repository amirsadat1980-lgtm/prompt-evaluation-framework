"""Load prompt versions and test cases from the examples/ manifest files."""

from __future__ import annotations

import json
from pathlib import Path

from .backends import PromptConstraints
from .evaluator import PromptVersion, EvalCase


def load_prompt_versions(manifest_path: Path) -> list[PromptVersion]:
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    versions = []
    for entry in entries:
        text_path = manifest_path.parent / entry["file"]
        constraints = PromptConstraints(**entry.get("constraints", {}))
        versions.append(
            PromptVersion(
                id=entry["id"],
                text=text_path.read_text(encoding="utf-8").strip(),
                constraints=constraints,
            )
        )
    return versions


def load_test_cases(path: Path) -> list[EvalCase]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(id=e["id"], input_text=e["input_text"], key_facts=e.get("key_facts", []))
        for e in entries
    ]
