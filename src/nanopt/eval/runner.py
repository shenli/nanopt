"""Checkpoint-agnostic arithmetic evaluation with immediate example persistence."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from nanopt.data.schemas import ArithmeticTask
from nanopt.eval.metrics import aggregate_results
from nanopt.eval.records import EvaluationResult
from nanopt.eval.verifier import VerifierContractError, verify_task_response
from nanopt.models.renderer import ChatRenderer
from nanopt.rollout.sampler import GenerationResult, SamplingConfig
from nanopt.runtime.artifacts import append_jsonl, canonical_json, sha256_bytes, write_json


class EvaluationBackend(Protocol):
    """Minimal checkpoint-independent surface required by the evaluation loop."""

    def render_prompt(self, prompt: str) -> Sequence[int]: ...

    def generate(
        self, prompt_token_ids: Sequence[int], config: SamplingConfig, *, seed: int
    ) -> GenerationResult: ...

    def decode_completion(self, token_ids: Sequence[int]) -> str: ...


class DecoderTokenizer(Protocol):
    """Tokenizer decoding surface used after exact generated IDs have been saved."""

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True)
class EvaluationIdentity:
    """Run/checkpoint labels kept separate from model implementation details."""

    run_id: str
    checkpoint_id: str


@dataclass(frozen=True)
class EvaluationPlan:
    """One deterministic or sampled pass over a task collection."""

    sampling: SamplingConfig
    samples_per_task: int
    base_seed: int
    max_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if not self.sampling.do_sample and self.samples_per_task != 1:
            raise ValueError("deterministic evaluation must use exactly one sample per task")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")

    def fingerprint(self) -> str:
        value = {
            "max_new_tokens": self.sampling.max_new_tokens,
            "do_sample": self.sampling.do_sample,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "eos_token_id": self.sampling.eos_token_id,
            "samples_per_task": self.samples_per_task,
            "max_prompt_tokens": self.max_prompt_tokens,
            "seed_schedule": "sha256-task-sample-v1",
        }
        return sha256_bytes(canonical_json(value))


def evaluation_seed(base_seed: int, task_id: str, sample_index: int) -> int:
    """Derive a stable non-negative 63-bit seed from task identity and sample index."""

    if not task_id:
        raise ValueError("task_id must not be empty")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    value = f"evaluation-seed-v1\0{base_seed}\0{task_id}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def _result_id(
    identity: EvaluationIdentity,
    task_id: str,
    sample_index: int,
    generation_fingerprint: str,
) -> str:
    value = canonical_json(
        {
            "run_id": identity.run_id,
            "checkpoint_id": identity.checkpoint_id,
            "task_id": task_id,
            "sample_index": sample_index,
            "generation_config_sha256": generation_fingerprint,
        }
    )
    return f"eval_{sha256_bytes(value)[:24]}"


def evaluate_to_artifacts(
    tasks: Iterable[ArithmeticTask],
    backend: EvaluationBackend,
    identity: EvaluationIdentity,
    plan: EvaluationPlan,
    *,
    samples_path: Path,
    summary_path: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> list[EvaluationResult]:
    """Generate, persist, verify, then aggregate every arithmetic example.

    Each ``EvaluationResult`` is appended to ``samples_path`` before it contributes to the final
    aggregate. An interrupted run therefore retains every completed example. Checkpoint-specific
    loading stays behind ``EvaluationBackend``; the loop sees only token IDs and identity strings.
    """

    task_list = list(tasks)
    if not task_list:
        raise ValueError("at least one evaluation task is required")
    if samples_path.exists() and samples_path.stat().st_size:
        raise ValueError("samples_path must be new or empty to prevent mixed evaluation runs")
    fingerprint = plan.fingerprint()
    results: list[EvaluationResult] = []
    for task in task_list:
        if task.split is None:
            raise ValueError(f"evaluation task {task.task_id} must have an assigned split")
        prompt_ids = tuple(int(value) for value in backend.render_prompt(task.prompt))
        if not prompt_ids:
            raise ValueError(f"renderer returned an empty prompt for task {task.task_id}")
        if plan.max_prompt_tokens is not None and len(prompt_ids) > plan.max_prompt_tokens:
            raise ValueError(
                f"rendered prompt for task {task.task_id} has {len(prompt_ids)} tokens, "
                f"exceeding the configured maximum {plan.max_prompt_tokens}; M3 does not "
                "silently truncate prompts"
            )
        for sample_index in range(plan.samples_per_task):
            seed = evaluation_seed(plan.base_seed, task.task_id, sample_index)
            started = clock()
            generation = backend.generate(prompt_ids, plan.sampling, seed=seed)
            elapsed = max(0.0, clock() - started)
            response = backend.decode_completion(generation.generated_token_ids)
            try:
                verification = verify_task_response(task, response)
                parser_status: Literal["valid", "invalid", "error"] = (
                    "valid" if verification.parser.valid else "invalid"
                )
                verifier_status: Literal["correct", "incorrect", "not_run", "error"] = (
                    "correct" if verification.correct else "incorrect"
                )
                parsed_answer = verification.candidate_answer
                metadata: dict[str, str | int | float | bool | None] = {
                    "family": task.family,
                    "difficulty": task.difficulty,
                    "parser_detail": verification.parser.status,
                    "policy_logp_sum": sum(generation.policy_logps),
                    "behavior_logp_sum": sum(generation.behavior_logps),
                }
            except VerifierContractError as exc:
                parser_status = "error"
                verifier_status = "error"
                parsed_answer = None
                metadata = {"verifier_error": str(exc)}
            result = EvaluationResult(
                result_id=_result_id(identity, task.task_id, sample_index, fingerprint),
                run_id=identity.run_id,
                checkpoint_id=identity.checkpoint_id,
                task_id=task.task_id,
                split=task.split,
                sample_index=sample_index,
                seed=seed,
                generation_config_sha256=fingerprint,
                prompt_token_ids=list(generation.prompt_token_ids),
                completion_token_ids=list(generation.generated_token_ids),
                response_text=response,
                parser_status=parser_status,
                parsed_answer=parsed_answer,
                verifier_status=verifier_status,
                reward_components={"correctness": float(verifier_status == "correct")},
                finish_reason=generation.finish_reason,
                generation_seconds=elapsed,
                metadata=metadata,
            )
            append_jsonl(samples_path, result.model_dump(mode="json"))
            results.append(result)
    write_json(summary_path, aggregate_results(results))
    return results


class LocalModelBackend:
    """Thin adapter joining the M2 renderer/tokenizer to the explicit M3 sampler."""

    def __init__(self, model: Any, tokenizer: DecoderTokenizer, renderer: ChatRenderer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.renderer = renderer

    def render_prompt(self, prompt: str) -> Sequence[int]:
        rendered = self.renderer.render_prompt([{"role": "user", "content": prompt}])
        return rendered.input_ids

    def generate(
        self, prompt_token_ids: Sequence[int], config: SamplingConfig, *, seed: int
    ) -> GenerationResult:
        from nanopt.rollout.sampler import sample_autoregressive

        return sample_autoregressive(
            self.model,
            torch.tensor(prompt_token_ids, dtype=torch.long),
            config,
            seed=seed,
        )

    def decode_completion(self, token_ids: Sequence[int]) -> str:
        # Exact IDs, including EOS, are already stored in GenerationResult. Removing tokenizer
        # control tokens only for parser-facing text prevents EOS from looking like trailing user
        # content without ever decoding and re-tokenizing a training trajectory.
        value = self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        return value
