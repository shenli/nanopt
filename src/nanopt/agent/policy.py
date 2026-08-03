"""Scripted oracle, replay, and Qwen structured-action policy adapters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from nanopt.agent.records import (
    ACTION_ADAPTER,
    AgentObservation,
    AgentPolicyIdentity,
)
from nanopt.config.models import AgentEvaluationExperiment, ModelProfile
from nanopt.models.adapters import load_lora_adapter
from nanopt.models.loading import load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.rollout.sampler import SamplingConfig, sample_autoregressive
from nanopt.sft.checkpoint import sha256_directory


@dataclass(frozen=True)
class PolicyResponse:
    text: str
    token_ids: list[int] | None
    seconds: float


class AgentPolicy(Protocol):
    identity: AgentPolicyIdentity

    def respond(self, observation: AgentObservation) -> PolicyResponse: ...


class ScriptedOraclePolicy:
    """Apply the reviewed oracle diff through the same model-visible action protocol."""

    def __init__(self, patch: str) -> None:
        self.identity = AgentPolicyIdentity(
            name="scripted_oracle",
            version="1",
            checkpoint_id=None,
            generation={"deterministic": True},
        )
        self.responses = [
            json.dumps({"tool": "apply_patch", "arguments": {"patch": patch}}),
            json.dumps({"tool": "run_tests", "arguments": {}}),
            json.dumps(
                {"tool": "finish", "arguments": {"summary": "Applied fix and ran public tests."}}
            ),
        ]
        self.index = 0

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        del observation
        if self.index >= len(self.responses):
            raise RuntimeError("scripted oracle exhausted before environment termination")
        response = self.responses[self.index]
        self.index += 1
        return PolicyResponse(response, None, 0.0)


class ReplayPolicy:
    """Feed retained model responses back through a fresh environment reset."""

    def __init__(self, responses: list[str], identity: AgentPolicyIdentity) -> None:
        self.identity = identity
        self.responses = responses
        self.index = 0

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        del observation
        if self.index >= len(self.responses):
            raise RuntimeError("replay trajectory exhausted before termination")
        response = self.responses[self.index]
        self.index += 1
        return PolicyResponse(response, None, 0.0)


class QwenStructuredPolicy:
    """Generate one exact-token response per turn and require the typed JSON action protocol."""

    def __init__(
        self,
        model_profile: ModelProfile,
        experiment: AgentEvaluationExperiment,
        *,
        adapter_path: Path | None,
        adapter_name: str,
        local_files_only: bool,
        device: str,
    ) -> None:
        loaded = load_qwen3_base(model_profile, local_files_only=local_files_only)
        model = loaded.model
        checkpoint_id = loaded.model_revision
        adapter_sha: str | None = None
        if adapter_path is not None:
            model = load_lora_adapter(
                model,
                adapter_path,
                adapter_name=adapter_name,
                trainable=False,
            )
            adapter_sha = sha256_directory(adapter_path)
            checkpoint_id = adapter_sha
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = loaded.tokenizer
        self.renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=model_profile.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        self.experiment = experiment
        self.turn = 0
        self.identity = AgentPolicyIdentity(
            name="qwen_structured_action",
            version="1",
            checkpoint_id=checkpoint_id,
            generation={
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "adapter_sha256": adapter_sha,
                "max_new_tokens": experiment.policy.max_new_tokens_per_turn,
                "temperature": experiment.policy.temperature,
                "top_p": experiment.policy.top_p,
                "seed": experiment.seed,
                "exact_token_ids_saved": True,
            },
        )

    @staticmethod
    def _system_instruction() -> str:
        schema = json.dumps(ACTION_ADAPTER.json_schema(), sort_keys=True)
        return (
            "You are editing a tiny repository through allow-listed tools. Return exactly one JSON "
            "object and no prose. Arbitrary shell commands are unavailable. Never modify tests. "
            f"Your action must validate against this schema: {schema}"
        )

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        prompt = self.renderer.render_prompt(
            [
                {"role": "system", "content": self._system_instruction()},
                {
                    "role": "user",
                    "content": observation.model_dump_json(exclude_none=False),
                },
            ]
        )
        config = SamplingConfig(
            max_new_tokens=self.experiment.policy.max_new_tokens_per_turn,
            do_sample=True,
            temperature=self.experiment.policy.temperature,
            top_p=self.experiment.policy.top_p,
            eos_token_id=qwen_chat_terminator_id(self.tokenizer),
        )
        started = time.perf_counter()
        generation = sample_autoregressive(
            self.model,
            torch.tensor(prompt.input_ids, dtype=torch.long),
            config,
            seed=self.experiment.seed + self.turn,
        )
        seconds = time.perf_counter() - started
        self.turn += 1
        value = self.tokenizer.decode(
            list(generation.generated_token_ids), skip_special_tokens=True
        )
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        return PolicyResponse(value, list(generation.generated_token_ids), seconds)
