"""Scoring criteria.

Each scorer takes plain text (and, where useful, a bit of context) and
returns a float in [0, 1]. All scoring is explicit and rule-based so a
result can always be explained and reproduced — there is no hidden
"LLM judge" black box by default.
"""

from __future__ import annotations

from .backends import PromptConstraints
from .text_utils import (
    JARGON_SUBSTITUTIONS,
    count_syllables,
    jaccard_similarity,
    split_sentences,
    split_words,
    top_keywords,
)


def score_instruction_following(output: str, constraints: PromptConstraints) -> float:
    """Fraction of the prompt's declared structural constraints that are met."""
    checks: list[bool] = []

    if constraints.max_sentences is not None:
        checks.append(len(split_sentences(output)) <= constraints.max_sentences)

    if constraints.max_words is not None:
        checks.append(len(split_words(output)) <= constraints.max_words)

    if constraints.forbid_jargon:
        lowered = output.lower()
        checks.append(not any(term in lowered for term in JARGON_SUBSTITUTIONS))

    if constraints.must_start_with is not None:
        checks.append(output.lower().startswith(constraints.must_start_with.lower()))

    if not checks:
        return 1.0  # No constraints declared -> nothing to violate.
    return sum(checks) / len(checks)


def score_clarity(output: str) -> float:
    """Approximate readability via a from-scratch Flesch Reading Ease calculation.

    Score is normalized to [0, 1], where higher means easier to read for a
    general, non-technical audience.
    """
    sentences = split_sentences(output)
    words = split_words(output)
    if not sentences or not words:
        return 0.0

    syllables = sum(count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)

    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return max(0.0, min(1.0, flesch / 100))


def score_relevance(output: str, source_text: str, k: int = 8) -> float:
    """Fraction of the source's top keywords that appear in the output."""
    keywords = top_keywords(source_text, k=k)
    if not keywords:
        return 1.0
    lowered = output.lower()
    hits = sum(1 for kw in keywords if kw in lowered)
    return hits / len(keywords)


def score_accuracy(output: str, key_facts: list[str]) -> float:
    """Fraction of required key facts (short reference phrases) present in the output."""
    if not key_facts:
        return 1.0
    lowered = output.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in lowered)
    return hits / len(key_facts)


def score_consistency(outputs: list[str]) -> float:
    """Average pairwise word-overlap similarity across repeated-run outputs."""
    if len(outputs) < 2:
        return 1.0
    pairs = [
        jaccard_similarity(outputs[i], outputs[j])
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    ]
    return sum(pairs) / len(pairs)


CRITERIA = ("accuracy", "relevance", "clarity", "consistency", "instruction_following")
