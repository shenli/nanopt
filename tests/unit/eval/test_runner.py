from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from nanopt.data.arithmetic import generate_task
from nanopt.eval.runner import (
    EvaluationIdentity,
    EvaluationPlan,
    LocalModelBackend,
    evaluate_to_artifacts,
    evaluation_seed,
)
from nanopt.rollout.sampler import GenerationResult, SamplingConfig


class ExactAnswerBackend:
    def __init__(self, answers: dict[int, str]) -> None:
        self.answers = answers
        self.seeds: list[int] = []

    def render_prompt(self, prompt: str) -> Sequence[int]:
        return [len(prompt)]

    def generate(
        self, prompt_token_ids: Sequence[int], config: SamplingConfig, *, seed: int
    ) -> GenerationResult:
        self.seeds.append(seed)
        return GenerationResult(
            prompt_token_ids=tuple(prompt_token_ids),
            generated_token_ids=(seed % 1000,),
            active_mask=(True,),
            policy_logps=(-0.25,),
            behavior_logps=(-0.25,),
            finish_reason="eos",
        )

    def decode_completion(self, token_ids: Sequence[int]) -> str:
        return self.answers[token_ids[0]]


class RecordingTokenizer:
    def __init__(self) -> None:
        self.skip_special_tokens: bool | None = None

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.skip_special_tokens = skip_special_tokens
        return "<answer>1</answer>"


class TwoTokenPromptBackend(ExactAnswerBackend):
    def render_prompt(self, prompt: str) -> Sequence[int]:
        return [len(prompt), 1]


def test_runner_writes_each_example_before_summary(tmp_path: Path) -> None:
    task = generate_task(family="addition_subtraction", difficulty=1, seed=7).model_copy(
        update={"split": "test_iid"}
    )
    plan = EvaluationPlan(SamplingConfig(4, False, eos_token_id=2), 1, 42)
    seed = evaluation_seed(42, task.task_id, 0)
    backend = ExactAnswerBackend({seed % 1000: f"<answer>{task.target.canonical_answer}</answer>"})
    ticks = iter([10.0, 10.25])

    results = evaluate_to_artifacts(
        [task],
        backend,
        EvaluationIdentity("run-1", "base"),
        plan,
        samples_path=tmp_path / "samples.jsonl",
        summary_path=tmp_path / "summary.json",
        clock=lambda: next(ticks),
    )

    saved = json.loads((tmp_path / "samples.jsonl").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert results[0].verifier_status == "correct"
    assert saved["completion_token_ids"] == [seed % 1000]
    assert saved["generation_seconds"] == 0.25
    assert summary["accuracy"]["estimate"] == 1.0


def test_seed_schedule_and_results_are_reproducible(tmp_path: Path) -> None:
    task = generate_task(family="multiplication", difficulty=1, seed=2).model_copy(
        update={"split": "test_iid"}
    )
    plan = EvaluationPlan(SamplingConfig(2, True), 2, -5)
    seeds = [evaluation_seed(-5, task.task_id, index) for index in range(2)]
    responses = {seed % 1000: f"<answer>{task.target.canonical_answer}</answer>" for seed in seeds}
    first_backend = ExactAnswerBackend(responses)
    second_backend = ExactAnswerBackend(responses)

    first = evaluate_to_artifacts(
        [task],
        first_backend,
        EvaluationIdentity("run", "base"),
        plan,
        samples_path=tmp_path / "first.jsonl",
        summary_path=tmp_path / "first-summary.json",
        clock=lambda: 0.0,
    )
    second = evaluate_to_artifacts(
        [task],
        second_backend,
        EvaluationIdentity("run", "base"),
        plan,
        samples_path=tmp_path / "second.jsonl",
        summary_path=tmp_path / "second-summary.json",
        clock=lambda: 0.0,
    )
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert first_backend.seeds == seeds


def test_evaluation_plan_and_runner_reject_ambiguous_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EvaluationPlan(SamplingConfig(1, False), 2, 0)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_to_artifacts(
            [],
            ExactAnswerBackend({}),
            EvaluationIdentity("run", "base"),
            EvaluationPlan(SamplingConfig(1, False), 1, 0),
            samples_path=tmp_path / "samples.jsonl",
            summary_path=tmp_path / "summary.json",
        )
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        EvaluationPlan(SamplingConfig(1, False), 1, 0, max_prompt_tokens=0)


def test_runner_rejects_prompt_over_profile_limit(tmp_path: Path) -> None:
    task = generate_task(family="multiplication", difficulty=1, seed=12).model_copy(
        update={"split": "test_iid"}
    )
    with pytest.raises(ValueError, match="does not silently truncate"):
        evaluate_to_artifacts(
            [task],
            TwoTokenPromptBackend({}),
            EvaluationIdentity("run", "base"),
            EvaluationPlan(SamplingConfig(1, False), 1, 0, max_prompt_tokens=1),
            samples_path=tmp_path / "samples.jsonl",
            summary_path=tmp_path / "summary.json",
        )


def test_local_backend_drops_control_tokens_only_from_parser_text() -> None:
    tokenizer = RecordingTokenizer()
    backend = LocalModelBackend(object(), tokenizer, object())  # type: ignore[arg-type]
    assert backend.decode_completion([5, 2]) == "<answer>1</answer>"
    assert tokenizer.skip_special_tokens is True
