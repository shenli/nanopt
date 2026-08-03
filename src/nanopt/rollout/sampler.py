"""A deliberately explicit token-at-a-time causal language-model sampler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

FinishReason = Literal["eos", "stop_sequence", "length"]


@dataclass(frozen=True)
class SamplingConfig:
    """All choices that change the distribution of generated tokens.

    ``temperature`` and ``top_p`` are applied only when ``do_sample`` is true. The NanoPT
    reference policy uses temperature 1 and top-p 1, so behavior and raw-policy probabilities are
    identical. Other values remain supported for evaluation and are recorded explicitly.
    """

    max_new_tokens: int
    do_sample: bool
    temperature: float = 1.0
    top_p: float = 1.0
    eos_token_id: int | None = None
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
        if any(not sequence for sequence in self.stop_token_sequences):
            raise ValueError("stop token sequences must not be empty")
        if any(token < 0 for sequence in self.stop_token_sequences for token in sequence):
            raise ValueError("stop token IDs must be non-negative")
        if len(set(self.stop_token_sequences)) != len(self.stop_token_sequences):
            raise ValueError("stop token sequences must not contain duplicates")


@dataclass(frozen=True)
class GenerationResult:
    """Exact sampled-token evidence for one prompt.

    Token ID tuples are in tokenizer coordinates. Both log-probability tuples are FP32 values with
    length ``generated_token_ids``: ``policy_logps`` use the model's unmodified softmax, while
    ``behavior_logps`` include sampling temperature and top-p renormalization in sampled mode. In
    greedy mode they deliberately copy the raw-policy diagnostic rather than claim a stochastic
    behavior probability. ``active_mask`` includes a generated EOS.
    """

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    active_mask: tuple[bool, ...]
    policy_logps: tuple[float, ...]
    behavior_logps: tuple[float, ...]
    finish_reason: FinishReason


def _extract_logits(output: Any) -> Tensor:
    logits = (
        output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    )
    if not isinstance(logits, Tensor):
        raise TypeError("model output must expose a tensor named 'logits'")
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            f"model logits must have shape [1, sequence, vocabulary], got {tuple(logits.shape)}"
        )
    if logits.shape[1] == 0 or logits.shape[2] == 0:
        raise ValueError("model logits must have non-empty sequence and vocabulary dimensions")
    if not logits.is_floating_point():
        raise TypeError(f"model logits must be floating point, got {logits.dtype}")
    return logits


def _top_p_log_probs(logits: Tensor, top_p: float) -> Tensor:
    """Return normalized log probabilities after nucleus filtering one vocabulary row."""

    log_probs = F.log_softmax(logits.float(), dim=-1)
    if top_p == 1.0:
        return log_probs
    sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
    sorted_probs = sorted_log_probs.exp()
    # Keep token j when the probability mass strictly before j is below top_p. This always keeps
    # the most likely token and also keeps the first token that crosses the threshold.
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    keep_sorted = mass_before < top_p
    filtered_sorted = sorted_log_probs.masked_fill(~keep_sorted, -torch.inf)
    filtered = torch.full_like(filtered_sorted, -torch.inf)
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)
    return F.log_softmax(filtered, dim=-1)


def _model_device(model: Any) -> torch.device:
    try:
        return torch.device(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def sample_autoregressive(
    model: Any,
    prompt_token_ids: Tensor,
    config: SamplingConfig,
    *,
    seed: int = 0,
) -> GenerationResult:
    """Generate one completion while preserving sampled IDs and their exact log probabilities.

    Args:
        model: A causal model called as ``model(input_ids=..., attention_mask=...)`` whose output
            exposes logits shaped ``[1, sequence, vocabulary]``. No ``generate`` helper is used.
        prompt_token_ids: One-dimensional integer tensor ``[prompt_sequence]``. Batching is left to
            the evaluation runner so variable-length prompt and seed behavior remain obvious.
        config: Sampling and termination choices.
        seed: Seed for a private ``torch.Generator``. Global PyTorch RNG state is not consumed.

    Returns:
        :class:`GenerationResult` with exact IDs, masks, raw-policy log probabilities, sampling-
        behavior log probabilities, and the finish reason.

    The implementation recomputes the full prefix each step. That is intentionally slow but easy
    to audit; a future cached backend must pass parity tests against this reference implementation.
    Inference is FP32 at the softmax boundary even if model logits use BF16/FP16.
    """

    if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
        raise ValueError("prompt_token_ids must have shape [prompt_sequence] and be non-empty")
    if prompt_token_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("prompt_token_ids must have dtype int32 or int64")

    device = _model_device(model)
    sequence = prompt_token_ids.to(device=device, dtype=torch.int64).unsqueeze(0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated: list[int] = []
    policy_logps: list[float] = []
    behavior_logps: list[float] = []
    finish_reason: FinishReason = "length"
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                attention_mask = torch.ones_like(sequence, dtype=torch.long)
                logits = _extract_logits(model(input_ids=sequence, attention_mask=attention_mask))[
                    0, -1
                ]
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("next-token logits contain NaN or infinity")
                raw_log_probs = F.log_softmax(logits.float(), dim=-1)
                if config.do_sample:
                    behavior = _top_p_log_probs(logits.float() / config.temperature, config.top_p)
                    next_token = torch.multinomial(
                        behavior.exp(), num_samples=1, generator=generator
                    )
                else:
                    behavior = raw_log_probs
                    next_token = raw_log_probs.argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                generated.append(token_id)
                policy_logps.append(float(raw_log_probs[token_id].item()))
                behavior_logps.append(float(behavior[token_id].item()))
                sequence = torch.cat((sequence, next_token.reshape(1, 1)), dim=1)
                if config.eos_token_id is not None and token_id == config.eos_token_id:
                    finish_reason = "eos"
                    break
                if any(
                    len(generated) >= len(stop) and tuple(generated[-len(stop) :]) == stop
                    for stop in config.stop_token_sequences
                ):
                    finish_reason = "stop_sequence"
                    break
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    return GenerationResult(
        prompt_token_ids=tuple(int(value) for value in prompt_token_ids.tolist()),
        generated_token_ids=tuple(generated),
        active_mask=tuple(True for _ in generated),
        policy_logps=tuple(policy_logps),
        behavior_logps=tuple(behavior_logps),
        finish_reason=finish_reason,
    )
