from pef.backends import MockBackend, PromptConstraints
from pef.text_utils import split_sentences, split_words

INPUT_TEXT = (
    "Large language models answer questions by running inference across a distributed "
    "system of GPU servers. When traffic spikes, inference latency increases because "
    "requests queue up while GPUs are busy. Engineers reduce this bottleneck with "
    "quantization, a technique that shrinks a model's memory footprint so it runs faster. "
    "Another common fix is dynamic batching, which groups multiple requests together so "
    "the hardware is used more efficiently. Even with these optimizations, a single "
    "complex query can still take several seconds to return a full response. Teams "
    "continuously monitor throughput and latency to decide when it's time to add more servers."
)


def test_mock_backend_respects_max_sentences():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=2)
    output = backend.generate("summarize", constraints, INPUT_TEXT, prompt_id="p", input_id="i")
    assert len(split_sentences(output)) <= 2


def test_mock_backend_respects_max_words():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=5, max_words=10)
    output = backend.generate("summarize", constraints, INPUT_TEXT, prompt_id="p", input_id="i")
    assert len(split_words(output)) <= 10


def test_mock_backend_strips_jargon_when_forbidden():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=5, forbid_jargon=True)
    output = backend.generate("summarize", constraints, INPUT_TEXT, prompt_id="p", input_id="i")
    assert "quantization" not in output.lower()


def test_mock_backend_keeps_jargon_when_allowed():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=5, forbid_jargon=False)
    output = backend.generate("summarize", constraints, INPUT_TEXT, prompt_id="p", input_id="i")
    assert "quantization" in output.lower()


def test_mock_backend_honors_must_start_with():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=2, must_start_with="In short,")
    output = backend.generate("summarize", constraints, INPUT_TEXT, prompt_id="p", input_id="i")
    assert output.startswith("In short,")


def test_mock_backend_is_deterministic_for_the_primary_run():
    backend = MockBackend()
    constraints = PromptConstraints(max_sentences=2, forbid_jargon=True)
    out_a = backend.generate("summarize", constraints, INPUT_TEXT, run_index=0, prompt_id="p", input_id="i")
    out_b = backend.generate("summarize", constraints, INPUT_TEXT, run_index=0, prompt_id="p", input_id="i")
    assert out_a == out_b


def test_mock_backend_empty_input_returns_empty_string():
    backend = MockBackend()
    assert backend.generate("summarize", PromptConstraints(), "", prompt_id="p", input_id="i") == ""
