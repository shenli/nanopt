"""Scripted oracle, replay, and Qwen structured-action policy adapters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch

from nanopt.agent.context import agent_system_instruction, observation_text
from nanopt.agent.records import AgentObservation, AgentPolicyIdentity
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
    """Inspect, apply, test, and finish through the model-visible action protocol.

    The oracle is intentionally a demonstration rather than a shortcut. Its first two turns show
    how an agent discovers and reads the file before changing it, which gives Agent SFT examples
    for both read-only and mutating tools.
    """

    def __init__(self, patch: str) -> None:
        self.identity = AgentPolicyIdentity(
            name="scripted_oracle",
            version="1",
            checkpoint_id=None,
            generation={"deterministic": True},
        )
        edited_path = self._edited_path(patch)
        self.responses = [
            json.dumps({"tool": "list_files", "arguments": {"path": ".", "max_depth": 3}}),
            json.dumps(
                {
                    "tool": "read_file",
                    "arguments": {"path": edited_path, "start_line": 1, "end_line": 200},
                }
            ),
            json.dumps({"tool": "apply_patch", "arguments": {"patch": patch}}),
            json.dumps({"tool": "run_tests", "arguments": {}}),
            json.dumps(
                {"tool": "finish", "arguments": {"summary": "Applied fix and ran public tests."}}
            ),
        ]
        self.index = 0

    @staticmethod
    def _edited_path(patch: str) -> str:
        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                return line.removeprefix("+++ b/")
        raise ValueError("oracle patch must contain a +++ b/<path> header")

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        del observation
        if self.index >= len(self.responses):
            raise RuntimeError("scripted oracle exhausted before environment termination")
        response = self.responses[self.index]
        self.index += 1
        return PolicyResponse(response, None, 0.0)


class RecoveryOraclePolicy(ScriptedOraclePolicy):
    """Demonstrate recovery after the environment rejects one malformed action."""

    def __init__(self, patch: str) -> None:
        super().__init__(patch)
        self.identity = AgentPolicyIdentity(
            name="recovery_oracle",
            version="1",
            checkpoint_id=None,
            generation={"deterministic": True, "contains_invalid_prefix": True},
        )
        # This is deliberately invalid JSON, not a privileged operation. The environment records
        # the rejection and the next target demonstrates how to continue from that observation.
        self.responses.insert(0, "this is not a JSON action")


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
        self._conversation: list[dict[str, str]] = []
        self._previous_response: str | None = None
        self.identity = AgentPolicyIdentity(
            name="qwen_structured_action",
            version="1",
            checkpoint_id=checkpoint_id,
            generation={
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "adapter_sha256": adapter_sha,
                "max_new_tokens": experiment.policy.max_new_tokens_per_turn,
                "do_sample": experiment.policy.do_sample,
                "temperature": experiment.policy.temperature,
                "top_p": experiment.policy.top_p,
                "seed": experiment.seed,
                "exact_token_ids_saved": True,
                "stop_on_complete_json": True,
                "context_policy": experiment.policy.context_policy,
            },
        )

    @staticmethod
    def _system_instruction() -> str:
        return agent_system_instruction()

    @staticmethod
    def _observation_text(
        observation: AgentObservation,
        *,
        include_transcript: bool,
    ) -> str:
        return observation_text(observation, include_transcript=include_transcript)

    def _messages(self, observation: AgentObservation) -> list[dict[str, str]]:
        """Build either a snapshot prompt or a true alternating multi-turn conversation."""

        policy: Literal["observation_snapshot", "full_transcript"] = (
            self.experiment.policy.context_policy
        )
        if policy == "observation_snapshot":
            return [
                {"role": "system", "content": self._system_instruction()},
                {
                    "role": "user",
                    "content": self._observation_text(observation, include_transcript=True),
                },
            ]

        if not observation.transcript:
            self._conversation = [
                {"role": "system", "content": self._system_instruction()},
                {
                    "role": "user",
                    "content": self._observation_text(observation, include_transcript=False),
                },
            ]
            self._previous_response = None
        else:
            if self._previous_response is None or not self._conversation:
                raise RuntimeError("full-transcript policy lost its preceding model response")
            self._conversation.extend(
                [
                    {"role": "assistant", "content": self._previous_response},
                    {
                        "role": "user",
                        "content": self._observation_text(observation, include_transcript=False),
                    },
                ]
            )
            self._previous_response = None
        return list(self._conversation)

    def respond(self, observation: AgentObservation) -> PolicyResponse:
        prompt = self.renderer.render_prompt(self._messages(observation))
        config = SamplingConfig(
            max_new_tokens=self.experiment.policy.max_new_tokens_per_turn,
            do_sample=self.experiment.policy.do_sample,
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
            stop_predicate=self._is_complete_json_action,
        )
        seconds = time.perf_counter() - started
        self.turn += 1
        value = self.tokenizer.decode(
            list(generation.generated_token_ids), skip_special_tokens=True
        )
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        if self.experiment.policy.context_policy == "full_transcript":
            self._previous_response = value
        return PolicyResponse(value, list(generation.generated_token_ids), seconds)

    def _is_complete_json_action(self, token_ids: tuple[int, ...]) -> bool:
        """Stop at one complete JSON object without accepting or repairing invalid syntax."""

        value = self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
        if not isinstance(value, str):
            raise TypeError("tokenizer decode must return text")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict)
