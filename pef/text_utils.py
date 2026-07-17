"""Small, dependency-free text-processing helpers shared by scorers and backends."""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")

STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with without into onto from
    is are was were be been being this that these those it its it's as by
    not no do does did done have has had having will would can could should
    may might must shall than then so such very more most other some any
    all each every both few many much own same just also about above after
    again against below between during out over under while you your yours
    we our ours they their theirs he she his her him them i my me
    """.split()
)

# Terms our example prompts intentionally treat as "jargon" that a
# non-technical summary should avoid or translate into plain language.
# Replacements are deliberately kept shorter/simpler than the original term —
# a real plain-language rewrite should reduce reading difficulty, not just
# swap one technical phrase for a longer descriptive one.
JARGON_SUBSTITUTIONS = {
    "llm": "AI model",
    "llms": "AI models",
    "inference latency": "wait time",
    "distributed system": "many computers",
    "distributed systems": "many computers",
    "orchestration": "management",
    "asynchronous": "flexible",
    "idempotent": "safe to repeat",
    "vector embeddings": "meaning codes",
    "embeddings": "meaning codes",
    "throughput": "speed",
    "fine-tuning": "extra training",
    "quantization": "shrinking",
    "hyperparameters": "settings",
}


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation. Good enough for short, clean prose."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate to at most `max_words` words, using the same tokenization as
    split_words() so callers that check the result's word count agree with
    however it was produced."""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= max_words:
        return text
    end = matches[max_words - 1].end()
    return text[:end].rstrip().rstrip(",.;:") + "..."


def count_syllables(word: str) -> int:
    """Approximate syllable count via vowel-group counting (no external deps)."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def top_keywords(text: str, k: int = 8) -> list[str]:
    """Return the k most frequent non-stopword tokens (length > 2), by frequency."""
    words = [w.lower() for w in split_words(text) if len(w) > 2]
    freq: dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


def jaccard_similarity(a: str, b: str) -> float:
    set_a = {w.lower() for w in split_words(a)}
    set_b = {w.lower() for w in split_words(b)}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union
