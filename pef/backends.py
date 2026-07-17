"""Model backends.

A "backend" turns (prompt constraints, input text) into generated text.
The framework ships with a deterministic MockBackend so the whole
evaluation pipeline can be run and tested offline, with no API key and
reproducible results. A thin OpenAIBackend is included for anyone who
wants to point the same evaluation harness at a real model later.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Protocol

from .text_utils import JARGON_SUBSTITUTIONS, split_sentences, split_words, truncate_to_words, STOPWORDS


@dataclass
class PromptConstraints:
    """Structural instructions extracted from a prompt version.

    These are declared explicitly (see examples/prompts/prompts.json) rather
    than parsed out of free-form prompt text, so scoring stays predictable.
    """

    max_sentences: int | None = None
    max_words: int | None = None
    forbid_jargon: bool = False
    must_start_with: str | None = None


class Backend(Protocol):
    def generate(
        self, prompt_text: str, constraints: PromptConstraints, input_text: str, run_index: int = 0
    ) -> str: ...


def _sentence_importance(sentence: str, doc_freq: dict[str, int]) -> float:
    words = [w.lower() for w in split_words(sentence) if w.lower() not in STOPWORDS]
    if not words:
        return 0.0
    return sum(doc_freq.get(w, 0) for w in words) / len(words)


def _apply_jargon_substitutions(text: str) -> str:
    for term, replacement in JARGON_SUBSTITUTIONS.items():
        # Case-insensitive whole-phrase replacement, preserving surrounding text.
        lowered = text.lower()
        idx = lowered.find(term)
        while idx != -1:
            text = text[:idx] + replacement + text[idx + len(term) :]
            lowered = text.lower()
            idx = lowered.find(term, idx + len(replacement))
    return text


@dataclass
class MockBackend:
    """Deterministic, offline stand-in for a real LLM call.

    Performs simple extractive summarization (frequency-weighted sentence
    selection, a well-known baseline technique) and applies the prompt's
    declared constraints. Loosely-specified prompts (fewer/looser
    constraints) are given a higher chance of sampling variance across
    repeated runs, mirroring how vague instructions produce less
    consistent output from a real model.
    """

    base_variance: float = 0.35
    _rng_cache: dict[str, random.Random] = field(default_factory=dict, repr=False)

    def _rng_for(self, prompt_id: str, input_id: str, run_index: int) -> random.Random:
        key = f"{prompt_id}:{input_id}:{run_index}"
        if key not in self._rng_cache:
            seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
            self._rng_cache[key] = random.Random(seed)
        return self._rng_cache[key]

    def generate(
        self,
        prompt_text: str,
        constraints: PromptConstraints,
        input_text: str,
        run_index: int = 0,
        prompt_id: str = "prompt",
        input_id: str = "input",
    ) -> str:
        sentences = split_sentences(input_text)
        if not sentences:
            return ""

        doc_freq: dict[str, int] = {}
        for s in sentences:
            for w in split_words(s):
                w = w.lower()
                if w not in STOPWORDS:
                    doc_freq[w] = doc_freq.get(w, 0) + 1

        max_sentences = constraints.max_sentences or min(3, len(sentences))

        # Looser prompts (no explicit constraints at all) get more variance,
        # simulating how vague instructions yield less repeatable output.
        has_explicit_constraints = any(
            [constraints.max_sentences, constraints.max_words, constraints.forbid_jargon, constraints.must_start_with]
        )
        variance = self.base_variance if not has_explicit_constraints else self.base_variance * 0.25
        rng = self._rng_for(prompt_id, input_id, run_index)

        ranked = sorted(
            range(len(sentences)),
            key=lambda i: (-_sentence_importance(sentences[i], doc_freq), i),
        )
        chosen_idx = sorted(ranked[:max_sentences])

        if run_index > 0 and len(sentences) > max_sentences and rng.random() < variance:
            # Swap the weakest chosen sentence for the next-best alternative.
            alt_pool = [i for i in ranked[max_sentences : max_sentences + 2] if i not in chosen_idx]
            if alt_pool:
                chosen_idx[-1] = alt_pool[0]
                chosen_idx = sorted(chosen_idx)

        if run_index > 0 and len(chosen_idx) > 1 and rng.random() < variance:
            # Occasionally reorder two adjacent picks (sampling-order noise).
            i = rng.randrange(len(chosen_idx) - 1)
            chosen_idx[i], chosen_idx[i + 1] = chosen_idx[i + 1], chosen_idx[i]

        picked = [sentences[i] for i in chosen_idx]
        if constraints.forbid_jargon:
            picked = [_apply_jargon_substitutions(s) for s in picked]

        if constraints.max_words:
            # A well-behaved model respects a word budget by leaving out a
            # whole trailing sentence, not by cutting one off mid-thought:
            # drop the last-selected sentence(s) until it fits, reserving
            # room for the required prefix if there is one.
            budget = constraints.max_words - len(
                split_words(constraints.must_start_with or "")
            )
            while len(picked) > 1 and len(split_words(" ".join(picked))) > budget:
                picked.pop()
            if len(split_words(" ".join(picked))) > budget:
                picked = [truncate_to_words(picked[0], max(budget, 1))]

        output = " ".join(picked)

        if constraints.must_start_with and not output.lower().startswith(
            constraints.must_start_with.lower()
        ):
            output = f"{constraints.must_start_with} {output[0].lower()}{output[1:]}" if output else constraints.must_start_with

        return output.strip()


class OpenAIBackend:
    """Optional real-model backend. Not used unless explicitly selected.

    Requires the `openai` package and an OPENAI_API_KEY environment
    variable. Kept separate so the core framework has zero required
    third-party dependencies and can always be tested offline.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def generate(
        self, prompt_text: str, constraints: PromptConstraints, input_text: str, run_index: int = 0, **_: object
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError(
                "OpenAIBackend requires the 'openai' package. Install it with `pip install openai`."
            ) from exc

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": input_text},
            ],
            temperature=0.7 if run_index > 0 else 0.2,
        )
        return response.choices[0].message.content or ""
