"""CPU unit tests for the Megatron inference tokenizer adapter (issue #400).

These cover :class:`HFTokenizerInferenceAdapter` and ``_as_inference_tokenizer``
without importing Megatron, so they run in the CPU CI stage. The GPU end-to-end
generation path is covered by ``tests/fast-gpu/test_megatron_inference.py``.
"""

import pytest

from miles.backends.megatron_utils.inference import HFTokenizerInferenceAdapter, _as_inference_tokenizer


class _StubHFTokenizer:
    """Minimal stand-in for a HuggingFace ``PreTrainedTokenizerBase``."""

    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 100

    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append((text, add_special_tokens))
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids, skip_special_tokens=False):
        self.decode_calls.append((list(token_ids), skip_special_tokens))
        return " ".join(str(int(t)) for t in token_ids)


def test_tokenize_does_not_add_special_tokens():
    tok = _StubHFTokenizer()
    adapter = HFTokenizerInferenceAdapter(tok)

    out = adapter.tokenize("abc")

    assert out == [ord(c) % 100 for c in "abc"]
    # The controller adds BOS itself, so the adapter must not.
    assert tok.encode_calls == [("abc", False)]


def test_detokenize_forwards_skip_special_tokens():
    tok = _StubHFTokenizer()
    adapter = HFTokenizerInferenceAdapter(tok)

    assert adapter.detokenize([10, 11], skip_special_tokens=True) == "10 11"
    assert adapter.detokenize([10, 11], skip_special_tokens=False) == "10 11"
    assert tok.decode_calls == [([10, 11], True), ([10, 11], False)]


def test_bos_eos_eod_mapping():
    adapter = HFTokenizerInferenceAdapter(_StubHFTokenizer())
    assert adapter.bos == 1
    assert adapter.eos == 2
    assert adapter.eod == 2  # eod mirrors eos for termination
    assert adapter.vocab_size == 100


def test_bos_may_be_none():
    tok = _StubHFTokenizer()
    tok.bos_token_id = None
    adapter = HFTokenizerInferenceAdapter(tok)
    assert adapter.bos is None


def test_eod_raises_without_eos():
    tok = _StubHFTokenizer()
    tok.eos_token_id = None
    adapter = HFTokenizerInferenceAdapter(tok)
    with pytest.raises(ValueError, match="eos_token_id"):
        _ = adapter.eod


def test_as_inference_tokenizer_wraps_hf():
    tok = _StubHFTokenizer()
    wrapped = _as_inference_tokenizer(tok)
    assert isinstance(wrapped, HFTokenizerInferenceAdapter)


def test_as_inference_tokenizer_passes_through_megatron_style():
    adapter = HFTokenizerInferenceAdapter(_StubHFTokenizer())
    # Already Megatron-style (has tokenize/detokenize/bos/eod) -> returned as-is.
    assert _as_inference_tokenizer(adapter) is adapter


def test_as_inference_tokenizer_rejects_unsupported():
    class _Bad:
        pass

    with pytest.raises(TypeError, match="unsupported tokenizer"):
        _as_inference_tokenizer(_Bad())
