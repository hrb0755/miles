"""Drive a miles-built Megatron model as an inference engine (issue #400).

Miles trains with Megatron-LM but generates only through SGLang. This module
lets a model that miles built with Megatron-LM generate text directly, by
wiring it into Megatron-LM's own inference stack
(:mod:`megatron.core.inference`):

    GPTInferenceWrapper -> TextGenerationController -> StaticInferenceEngine

It is meant for testing / evaluating the Megatron path without standing up a
separate SGLang server.

Scope (v1): single GPU, tensor- and pipeline-parallel size 1, small models.
Distributed (TP/PP > 1) generation and an OpenAI-compatible endpoint are
deliberate follow-ups.

The heavy ``megatron.core.inference`` imports are deferred into the functions
that need them so that :class:`HFTokenizerInferenceAdapter` (pure Python) can be
imported and unit-tested without Megatron installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from megatron.core.inference.engines import StaticInferenceEngine
    from megatron.core.inference.inference_request import InferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.models.gpt import GPTModel


class HFTokenizerInferenceAdapter:
    """Expose a HuggingFace tokenizer through Megatron's inference interface.

    Megatron's ``TextGenerationController`` expects ``tokenize`` / ``detokenize``
    plus ``bos`` and ``eod`` attributes, whereas a HuggingFace
    ``PreTrainedTokenizerBase`` exposes ``encode`` / ``decode`` and
    ``bos_token_id`` / ``eos_token_id``. This thin adapter bridges the two so the
    Megatron model can be driven with the very tokenizer miles used for training.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer

    def tokenize(self, text: str) -> list[int]:
        # add_special_tokens=False: the controller adds BOS itself when needed.
        return self._tok.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def bos(self) -> int | None:
        return self._tok.bos_token_id

    @property
    def eos(self) -> int | None:
        return self._tok.eos_token_id

    @property
    def eod(self) -> int:
        """Termination token id required by the controller."""
        eos = self._tok.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id; cannot determine the termination id (eod) for generation")
        return eos

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size


def _as_inference_tokenizer(tokenizer: Any) -> Any:
    """Return a tokenizer that satisfies Megatron's inference interface.

    A Megatron-style tokenizer (has ``detokenize`` and ``eod``) is used as-is; a
    HuggingFace tokenizer (has ``encode`` / ``decode``) is wrapped.
    """
    if hasattr(tokenizer, "detokenize") and hasattr(tokenizer, "eod"):
        return tokenizer
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        return HFTokenizerInferenceAdapter(tokenizer)
    raise TypeError(
        f"unsupported tokenizer type {type(tokenizer)!r}: expected a Megatron "
        "tokenizer (tokenize/detokenize/bos/eod) or a HuggingFace tokenizer "
        "(encode/decode/bos_token_id/eos_token_id)"
    )


def _resolve_gpt_model(model: Any) -> GPTModel:
    """Extract the underlying ``GPTModel`` from what the actor holds.

    The Megatron actor holds ``self.model`` as a list of (DDP-wrapped) model
    chunks. v1 supports a single chunk (TP/PP = 1).
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.utils import unwrap_model

    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise NotImplementedError(
                f"Megatron inference v1 supports a single model chunk (TP/PP=1); "
                f"got {len(model)} chunks. Pipeline-parallel inference is a "
                "planned follow-up."
            )
        model = model[0]

    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, (list, tuple)):
        unwrapped = unwrapped[0]
    if not isinstance(unwrapped, GPTModel):
        raise TypeError(f"expected a megatron GPTModel, got {type(unwrapped)!r}")
    return unwrapped


def build_inference_engine(
    model: Any,
    tokenizer: Any,
    *,
    padded_vocab_size: int | None = None,
    params_dtype: torch.dtype | None = None,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_max_requests: int = 8,
    inference_max_seq_length: int = 2560,
    max_batch_size: int | None = None,
    random_seed: int | None = None,
    buffer_size_gb: float | None = None,
) -> StaticInferenceEngine:
    """Build a Megatron ``StaticInferenceEngine`` from a miles-built model.

    Args:
        model: The actor's model -- a ``GPTModel`` or a (one-element) list of
            (DDP-wrapped) chunks. It is unwrapped to the underlying ``GPTModel``.
        tokenizer: A HuggingFace tokenizer (e.g. miles' ``self.tokenizer``) or a
            Megatron-style tokenizer. HuggingFace tokenizers are adapted.
        padded_vocab_size: Final padded vocab size. Defaults to the model's own
            ``vocab_size``.
        params_dtype: Parameter dtype. Defaults to the model parameters' dtype.
        inference_batch_times_seqlen_threshold: Below this ``batch * seqlen`` the
            batch is not pipelined.
        inference_max_requests: Max concurrent requests (sizes CUDA graphs).
        inference_max_seq_length: Max sequence length (sizes CUDA graphs).
        max_batch_size: Optional engine-level batch cap.
        random_seed: Optional engine RNG seed for reproducible sampling.
        buffer_size_gb: KV-cache buffer size; ``None`` uses Megatron's default
            (large). Pass a small value for tests on shared hardware.

    Returns:
        A ready ``StaticInferenceEngine``; drive it with :func:`generate`.
    """
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.engines import StaticInferenceEngine
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
    from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
    from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController

    gpt_model = _resolve_gpt_model(model)
    config = gpt_model.config

    if padded_vocab_size is None:
        padded_vocab_size = gpt_model.vocab_size
    if params_dtype is None:
        params_dtype = next(gpt_model.parameters()).dtype

    wrapper_config = InferenceWrapperConfig(
        hidden_size=config.hidden_size,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
        padded_vocab_size=padded_vocab_size,
        inference_max_requests=inference_max_requests,
        inference_max_seq_length=inference_max_seq_length,
    )
    inference_context = StaticInferenceContext.from_config(wrapper_config)
    wrapped_model = GPTInferenceWrapper(gpt_model, wrapper_config, inference_context)
    controller = TextGenerationController(
        inference_wrapped_model=wrapped_model,
        tokenizer=_as_inference_tokenizer(tokenizer),
    )

    engine_kwargs: dict[str, Any] = {}
    if max_batch_size is not None:
        engine_kwargs["max_batch_size"] = max_batch_size
    if random_seed is not None:
        engine_kwargs["random_seed"] = random_seed
    if buffer_size_gb is not None:
        engine_kwargs["buffer_size_gb"] = buffer_size_gb
    return StaticInferenceEngine(controller, **engine_kwargs)


def generate(
    engine: StaticInferenceEngine,
    prompts: list[str],
    sampling_params: SamplingParams | None = None,
    **sampling_kwargs: Any,
) -> list[InferenceRequest]:
    """Generate from a Megatron inference engine.

    Args:
        engine: An engine from :func:`build_inference_engine`.
        prompts: Prompt strings.
        sampling_params: Explicit ``SamplingParams``; if ``None`` one is built
            from ``sampling_kwargs`` (e.g. ``num_tokens_to_generate``,
            ``temperature``, ``top_k``, ``top_p``).

    Returns:
        One ``InferenceRequest`` per prompt, each with ``generated_tokens`` and
        ``generated_text``.
    """
    if sampling_params is None:
        from megatron.core.inference.sampling_params import SamplingParams

        sampling_params = SamplingParams(**sampling_kwargs)
    return engine.generate(prompts=prompts, sampling_params=sampling_params)
