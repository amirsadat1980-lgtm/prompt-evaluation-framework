from pef.backends import PromptConstraints
from pef.criteria import (
    score_accuracy,
    score_clarity,
    score_consistency,
    score_instruction_following,
    score_relevance,
)


def test_instruction_following_all_satisfied():
    constraints = PromptConstraints(max_sentences=2, max_words=10, forbid_jargon=True, must_start_with="In short,")
    output = "In short, this works well."
    assert score_instruction_following(output, constraints) == 1.0


def test_instruction_following_partial():
    constraints = PromptConstraints(max_sentences=1, max_words=3, forbid_jargon=True, must_start_with="In short,")
    # Too many words, contains jargon, doesn't start with the required phrase, but is 1 sentence.
    output = "The LLM inference latency was high today unfortunately."
    score = score_instruction_following(output, constraints)
    assert 0.0 < score < 1.0


def test_instruction_following_no_constraints_defaults_to_full_score():
    assert score_instruction_following("anything at all", PromptConstraints()) == 1.0


def test_clarity_prefers_shorter_simpler_sentences():
    simple = "The cat sat on the mat. It was happy."
    complex_ = (
        "The feline organism, having situated itself upon the woven floor "
        "covering, subsequently exhibited indicators consistent with satisfaction."
    )
    assert score_clarity(simple) > score_clarity(complex_)


def test_clarity_empty_output_is_zero():
    assert score_clarity("") == 0.0


def test_relevance_counts_keyword_overlap():
    source = "Quantization reduces model size. Quantization also speeds up inference for large models."
    good_output = "This explains quantization and inference speed for large models."
    bad_output = "The weather today is sunny with a light breeze."
    assert score_relevance(good_output, source) > score_relevance(bad_output, source)


def test_accuracy_counts_key_fact_presence():
    key_facts = ["reduces hallucinations", "vector database"]
    full = "This approach reduces hallucinations by using a vector database for lookups."
    partial = "This approach reduces hallucinations somehow."
    none = "This is unrelated text."
    assert score_accuracy(full, key_facts) == 1.0
    assert score_accuracy(partial, key_facts) == 0.5
    assert score_accuracy(none, key_facts) == 0.0


def test_accuracy_with_no_key_facts_is_full_score():
    assert score_accuracy("anything", []) == 1.0


def test_consistency_identical_outputs_is_one():
    outputs = ["The model is fast.", "The model is fast."]
    assert score_consistency(outputs) == 1.0


def test_consistency_divergent_outputs_is_low():
    outputs = ["The model is fast.", "Bananas are yellow fruit."]
    assert score_consistency(outputs) < 0.3


def test_consistency_single_output_is_one():
    assert score_consistency(["only one output"]) == 1.0
