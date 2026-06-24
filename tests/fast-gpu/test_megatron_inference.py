"""Single-GPU end-to-end test for Megatron-as-inference-engine (issue #400).

Builds a tiny GPTModel in-process (random weights, TP=PP=1) and drives it
through ``miles.backends.megatron_utils.inference`` -- the same code path that
lets a miles-built Megatron model generate text without SGLang. Random weights
mean the token *values* are arbitrary; the test asserts the engine runs and
returns the requested number of in-range tokens.

Runs either way::

    pytest tests/fast-gpu/test_megatron_inference.py
    python3 tests/fast-gpu/test_megatron_inference.py   # bypasses conftest
"""

import os

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci

# Smallest GPU suite; the test itself uses a single GPU (cuda:0).
register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["megatron"])

VOCAB = 128
HIDDEN = 64
HEADS = 4
LAYERS = 2
MAX_SEQ = 64
GEN_TOKENS = 8


class _StubHFTokenizer:
    """Tiny HuggingFace-style tokenizer so the test exercises the real adapter.

    Token *values* are irrelevant for a random-weight model; this only needs the
    HF surface the adapter delegates to (encode/decode/bos_token_id/eos_token_id).
    """

    bos_token_id = None
    eos_token_id = VOCAB - 1
    vocab_size = VOCAB

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3, 4, 5]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in token_ids)


def _init_single_gpu_parallel():
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12399")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)


def _build_tiny_gpt_model():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        use_cpu_initialization=True,
    )
    return (
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            parallel_output=True,
            pre_process=True,
            post_process=True,
        )
        .cuda()
        .eval()
    )


def _teardown_parallel():
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_megatron_inference_generates_tokens():
    from miles.backends.megatron_utils.inference import build_inference_engine, generate

    _init_single_gpu_parallel()
    try:
        model = _build_tiny_gpt_model()
        engine = build_inference_engine(
            model,
            _StubHFTokenizer(),
            padded_vocab_size=VOCAB,
            inference_max_seq_length=MAX_SEQ,
            max_batch_size=4,
            buffer_size_gb=1.0,
        )
        # termination_id=-1 never matches a real token id, so generation runs the
        # full length and the count is deterministic.
        results = generate(
            engine,
            prompts=["1 2 3 4 5"],
            num_tokens_to_generate=GEN_TOKENS,
            termination_id=-1,
            temperature=1.0,
            top_k=1,
        )

        assert len(results) == 1
        gen = results[0].generated_tokens
        gen = gen.tolist() if hasattr(gen, "tolist") else list(gen)
        assert len(gen) == GEN_TOKENS
        assert all(0 <= t < VOCAB for t in gen)
        assert isinstance(results[0].generated_text, str)
    finally:
        _teardown_parallel()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    test_megatron_inference_generates_tokens()
    print("PASSED: Megatron drove generation as an inference engine.")
