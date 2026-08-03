"""Run the official recipe as explicit, hash-linked, independently resumable stages."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from nanopt.config.loader import ConfigRepository
from nanopt.config.models import BaseEvalExperiment
from nanopt.config.resolver import ResolutionResult, resolve_config
from nanopt.data.preferences import generate_preference_pairs
from nanopt.dpo.run import execute_dpo_run
from nanopt.eval.io import (
    read_arithmetic_tasks,
    read_split_manifest,
    validate_tasks_against_manifest,
)
from nanopt.eval.run import EvaluationMode, execute_evaluation_run, move_model
from nanopt.grpo.run import execute_grpo_run
from nanopt.models.loading import load_qwen3_base
from nanopt.pipeline.records import (
    FailureRetryRecord,
    PipelineManifest,
    PipelineStage,
    StageAttempt,
)
from nanopt.pipeline.report import build_pipeline_report
from nanopt.runtime.artifacts import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
    write_text,
)
from nanopt.runtime.environment import collect_git_metadata
from nanopt.runtime.run_context import RunContext, make_run_id, utc_now
from nanopt.sft.checkpoint import sha256_directory
from nanopt.sft.run import execute_sft_run

StageResult = tuple[Path | None, Path | None, str | None]
StageAction = Callable[[str], StageResult]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _resolved(
    repository: ConfigRepository,
    *,
    hardware: str,
    model: str,
    experiment: str,
    overrides: tuple[str, ...] = (),
) -> ResolutionResult:
    return resolve_config(
        repository=repository,
        hardware_id=hardware,
        model_id=model,
        experiment_id=experiment,
        overrides=overrides,
    )


def _release_accelerator_memory() -> None:
    """Make stage boundaries visible in memory measurements and avoid cross-stage retention."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attach_to_pipeline(
    context: RunContext,
    *,
    pipeline_run_id: str,
    parent_run_ids: list[str],
) -> None:
    """Link a completed ordinary run back to its parent before hashing its manifest."""

    context.manifest["pipeline_run_id"] = pipeline_run_id
    context.manifest["parent_run_ids"] = parent_run_ids
    write_json(context.manifest_path, context.manifest)


def _stage_specs() -> list[PipelineStage]:
    return [
        PipelineStage(id="load_calibration", kind="calibration"),
        PipelineStage(id="eval_calibration", kind="calibration"),
        PipelineStage(id="base_eval", kind="evaluation"),
        PipelineStage(id="sft_calibration", kind="calibration"),
        PipelineStage(id="sft", kind="training"),
        PipelineStage(id="sft_eval", kind="evaluation"),
        PipelineStage(id="preferences", kind="data"),
        PipelineStage(id="dpo_calibration", kind="calibration"),
        PipelineStage(id="dpo", kind="training"),
        PipelineStage(id="dpo_eval", kind="evaluation"),
        PipelineStage(id="grpo_calibration", kind="calibration"),
        PipelineStage(id="grpo", kind="training"),
        PipelineStage(id="grpo_eval", kind="evaluation"),
        PipelineStage(id="grpo_eval_repeat", kind="evaluation"),
        PipelineStage(id="report", kind="report"),
    ]


class PipelineRunner:
    """Mutable lifecycle facade that atomically validates every parent-manifest update."""

    def __init__(self, directory: Path, manifest: PipelineManifest) -> None:
        self.directory = directory
        self.manifest = manifest

    @property
    def manifest_path(self) -> Path:
        return self.directory / "pipeline_manifest.json"

    def write(self) -> None:
        validated = PipelineManifest.model_validate(self.manifest.model_dump(), strict=True)
        write_json(self.manifest_path, validated.model_dump(mode="json"))

    def stage(self, stage_id: str) -> PipelineStage:
        return next(stage for stage in self.manifest.stages if stage.id == stage_id)

    def completed_output(self, stage_id: str) -> Path | None:
        stage = self.stage(stage_id)
        return self.directory / stage.output_path if stage.output_path else None

    def _verify_completed(self, stage: PipelineStage) -> bool:
        if stage.status != "completed" or not stage.attempts:
            return False
        attempt = stage.attempts[-1]
        if attempt.child_manifest_sha256 is not None:
            if attempt.run_directory is None:
                return False
            child = self.directory / attempt.run_directory / "run_manifest.json"
            if not child.is_file() or sha256_file(child) != attempt.child_manifest_sha256:
                return False
        if stage.output_path is not None and stage.output_checkpoint_sha256 is not None:
            output = self.directory / stage.output_path
            if output.is_dir():
                return sha256_directory(output) == stage.output_checkpoint_sha256
            if output.is_file():
                return sha256_file(output) == stage.output_checkpoint_sha256
            return False
        return True

    def run_stage(
        self,
        stage_id: str,
        *,
        input_checkpoint_sha256: str | None,
        action: StageAction,
    ) -> StageResult:
        """Skip a verified completion or append a retained retry attempt."""

        stage = self.stage(stage_id)
        if self._verify_completed(stage):
            output = self.completed_output(stage_id)
            run_dir = (
                self.directory / stage.attempts[-1].run_directory
                if stage.attempts[-1].run_directory
                else None
            )
            return run_dir, output, stage.output_checkpoint_sha256
        if stage.status == "completed":
            raise ValueError(f"completed stage {stage_id!r} failed hash verification")
        if stage.input_checkpoint_sha256 not in {None, input_checkpoint_sha256}:
            raise ValueError(f"stage {stage_id!r} input checkpoint changed across resume")

        attempt_number = len(stage.attempts) + 1
        run_id = stage_id if attempt_number == 1 else f"{stage_id}-retry-{attempt_number}"
        stage.status = "running"
        stage.input_checkpoint_sha256 = input_checkpoint_sha256
        self.write()
        started_at = utc_now()
        started = time.monotonic()
        try:
            run_dir, output, output_sha = action(run_id)
            if run_dir is not None:
                manifest_sha = sha256_file(run_dir / "run_manifest.json")
                relative_run = run_dir.relative_to(self.directory).as_posix()
            else:
                manifest_sha = None
                relative_run = None
            artifact_sha = output_sha if run_dir is None else None
            stage.attempts.append(
                StageAttempt(
                    attempt=attempt_number,
                    run_id=run_id,
                    run_directory=relative_run,
                    child_manifest_sha256=manifest_sha,
                    artifact_sha256=artifact_sha,
                    started_at=started_at,
                    finished_at=utc_now(),
                    wall_seconds=time.monotonic() - started,
                    status="completed",
                )
            )
            stage.status = "completed"
            stage.output_checkpoint_sha256 = output_sha
            stage.output_path = output.relative_to(self.directory).as_posix() if output else None
            self.write()
            return run_dir, output, output_sha
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            stage.attempts.append(
                StageAttempt(
                    attempt=attempt_number,
                    run_id=run_id,
                    started_at=started_at,
                    finished_at=utc_now(),
                    wall_seconds=time.monotonic() - started,
                    status="failed",
                    failure=failure,
                )
            )
            stage.status = "failed"
            self.manifest.failures_and_retries.append(
                FailureRetryRecord(stage_id=stage_id, attempt=attempt_number, failure=failure)
            )
            self.manifest.status = "failed"
            self.manifest.finished_at = utc_now()
            self.write()
            raise
        finally:
            _release_accelerator_memory()


def _load_calibration(
    result: ResolutionResult,
    *,
    runs_root: Path,
    run_id: str,
    pipeline_run_id: str,
    local_files_only: bool,
    device: str,
) -> StageResult:
    """Measure a bare model load in a normal child run manifest."""

    from nanopt.runtime.run_context import create_run_context

    context = create_run_context(result, artifacts_root=runs_root, run_id=run_id)
    try:
        context.set_status("running")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        selected = move_model(loaded, device)
        peak = torch.cuda.max_memory_reserved() if selected == "cuda" else 0
        context.manifest["model"].update(
            {
                "resolved_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "base_parameter_count": loaded.parameters.total,
                "trainable_parameter_count": loaded.parameters.trainable,
            }
        )
        write_json(
            context.run_dir / "summary.json",
            {"device": selected, "peak_reserved_bytes": peak, "representative": False},
        )
        context.manifest["artifacts"] = [
            {
                "path": "summary.json",
                "kind": "load_calibration",
                "sha256": sha256_file(context.run_dir / "summary.json"),
            }
        ]
        context.set_status("completed")
        _attach_to_pipeline(context, pipeline_run_id=pipeline_run_id, parent_run_ids=[])
        return context.run_dir, None, None
    except Exception as exc:
        context.set_status(
            "failed", failure={"type": type(exc).__name__, "message": str(exc), "phase": "load"}
        )
        raise


def execute_pipeline(
    *,
    tasks_path: Path,
    artifacts_root: Path,
    recipe_id: str,
    config_dir: Path | None,
    run_id: str | None,
    resume: bool,
    local_files_only: bool,
    device: str,
) -> Path:
    """Execute or resume the frozen reference recipe without hidden trainer callbacks."""

    repository = ConfigRepository(config_dir) if config_dir else ConfigRepository()
    recipe = repository.recipe(recipe_id)
    expected_stage_ids = [stage.id for stage in _stage_specs()]
    configured_stage_ids = [stage.id for stage in recipe.stages]
    if configured_stage_ids != expected_stage_ids:
        raise ValueError(
            "the official pipeline recipe stages or order differ from the executable contract"
        )
    all_tasks = read_arithmetic_tasks(tasks_path)
    split_path = tasks_path.with_name("dataset_manifest.json")
    split_manifest = read_split_manifest(split_path)
    validate_tasks_against_manifest(all_tasks, split_manifest)
    selected_run_id = run_id or make_run_id("pipeline", recipe.model_dump(mode="json"))
    if Path(selected_run_id).name != selected_run_id or selected_run_id in {".", ".."}:
        raise ValueError("run_id must be a single path-safe component")
    pipeline_dir = artifacts_root / selected_run_id
    manifest_path = pipeline_dir / "pipeline_manifest.json"
    if resume:
        manifest = PipelineManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8"), strict=True
        )
        if manifest.task_file_sha256 != sha256_file(tasks_path):
            raise ValueError("task file changed since the pipeline began")
        if manifest.git["commit"] != collect_git_metadata()["commit"]:
            raise ValueError("source commit changed since the pipeline began")
        manifest.status = "running"
        manifest.finished_at = None
    else:
        pipeline_dir.mkdir(parents=True, exist_ok=False)
        (pipeline_dir / "runs").mkdir()
        (pipeline_dir / "data").mkdir()
        manifest = PipelineManifest(
            pipeline_run_id=selected_run_id,
            recipe_id=recipe.id,
            status="created",
            created_at=utc_now(),
            git=collect_git_metadata(),
            hardware_id=recipe.hardware,
            model_id=recipe.model,
            task_file_sha256=sha256_file(tasks_path),
            split_manifest_sha256=sha256_file(split_path),
            dataset_fingerprint=split_manifest.dataset_fingerprint,
            stages=_stage_specs(),
        )
    runner = PipelineRunner(pipeline_dir, manifest)
    runner.manifest.status = "running"
    runner.manifest.started_at = runner.manifest.started_at or utc_now()
    runner.write()
    runs_root = pipeline_dir / "runs"

    evaluation = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="base_eval",
    )
    if not isinstance(evaluation.config.experiment, BaseEvalExperiment):
        raise ValueError("base_eval must resolve to an evaluation profile")
    base_sha = sha256_bytes(
        canonical_json(
            {
                "model_id": evaluation.config.model.source.model_id,
                "revision": evaluation.config.model.source.revision,
                "tokenizer_revision": evaluation.config.model.source.tokenizer_revision,
            }
        )
    )

    runner.run_stage(
        "load_calibration",
        input_checkpoint_sha256=base_sha,
        action=lambda child_id: _load_calibration(
            evaluation,
            runs_root=runs_root,
            run_id=child_id,
            pipeline_run_id=selected_run_id,
            local_files_only=local_files_only,
            device=device,
        ),
    )

    def evaluate(
        child_id: str,
        *,
        checkpoint_id: str,
        adapter: Path | None,
        adapter_name: str,
        limit: int | None,
        parents: list[str],
    ) -> StageResult:
        context = execute_evaluation_run(
            evaluation,
            tasks_path=tasks_path,
            mode=EvaluationMode.deterministic,
            checkpoint_id=checkpoint_id,
            artifacts_root=runs_root,
            run_id=child_id,
            local_files_only=local_files_only,
            device=device,
            limit=limit,
            adapter_path=adapter,
            adapter_name=adapter_name,
        )
        _attach_to_pipeline(context, pipeline_run_id=selected_run_id, parent_run_ids=parents)
        return context.run_dir, None, None

    eval_calibration_dir, _, _ = runner.run_stage(
        "eval_calibration",
        input_checkpoint_sha256=base_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="base-calibration",
            adapter=None,
            adapter_name="base",
            limit=2,
            parents=[],
        ),
    )
    base_eval_dir, _, _ = runner.run_stage(
        "base_eval",
        input_checkpoint_sha256=base_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="base",
            adapter=None,
            adapter_name="base",
            limit=None,
            parents=[],
        ),
    )
    if eval_calibration_dir is None or base_eval_dir is None:
        raise RuntimeError("evaluation stages did not produce child runs")

    sft_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_sft",
    )
    sft_calibration_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_sft",
        overrides=(
            "training.micro_batch_size=1",
            "training.gradient_accumulation_steps=1",
            "training.max_steps=1",
        ),
    )

    def sft(child_id: str, *, calibration: bool) -> StageResult:
        context = execute_sft_run(
            sft_calibration_result if calibration else sft_result,
            tasks_path=tasks_path,
            artifacts_root=runs_root,
            run_id=child_id,
            local_files_only=local_files_only,
            device=device,
            train_limit=2 if calibration else None,
        )
        _attach_to_pipeline(context, pipeline_run_id=selected_run_id, parent_run_ids=[])
        adapter = context.run_dir / "adapter" / "sft"
        return context.run_dir, adapter, sha256_directory(adapter)

    runner.run_stage(
        "sft_calibration",
        input_checkpoint_sha256=base_sha,
        action=lambda child_id: sft(child_id, calibration=True),
    )
    sft_dir, sft_adapter, sft_sha = runner.run_stage(
        "sft",
        input_checkpoint_sha256=base_sha,
        action=lambda child_id: sft(child_id, calibration=False),
    )
    if sft_dir is None or sft_adapter is None or sft_sha is None:
        raise RuntimeError("SFT did not produce an adapter")
    sft_run_id = sft_dir.name
    sft_eval_dir, _, _ = runner.run_stage(
        "sft_eval",
        input_checkpoint_sha256=sft_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="sft",
            adapter=sft_adapter,
            adapter_name="sft",
            limit=None,
            parents=[sft_run_id],
        ),
    )
    if sft_eval_dir is None:
        raise RuntimeError("SFT evaluation did not produce a child run")

    preferences_path = pipeline_dir / "data" / "preferences.jsonl"
    audit_path = pipeline_dir / "data" / "preference_audit.json"

    def preferences(_child_id: str) -> StageResult:
        pairs, audit = generate_preference_pairs(
            all_tasks,
            source_dataset_fingerprint=split_manifest.dataset_fingerprint,
            seed=42,
        )
        lines = "".join(
            json.dumps(pair.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
            for pair in pairs
        )
        write_text(preferences_path, lines)
        write_json(audit_path, audit.model_dump(mode="json"))
        data_directory = preferences_path.parent
        return None, data_directory, sha256_directory(data_directory)

    runner.run_stage("preferences", input_checkpoint_sha256=None, action=preferences)

    dpo_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_dpo",
    )
    dpo_calibration_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_dpo",
        overrides=(
            "training.pair_micro_batch_size=1",
            "training.gradient_accumulation_steps=1",
            "training.epochs=1",
        ),
    )

    def dpo(child_id: str, *, calibration: bool) -> StageResult:
        context = execute_dpo_run(
            dpo_calibration_result if calibration else dpo_result,
            preferences_path=preferences_path,
            sft_adapter_path=sft_adapter,
            artifacts_root=runs_root,
            run_id=child_id,
            local_files_only=local_files_only,
            device=device,
            pair_limit=2 if calibration else None,
        )
        _attach_to_pipeline(context, pipeline_run_id=selected_run_id, parent_run_ids=[sft_run_id])
        adapter = context.run_dir / "adapter" / "dpo"
        return context.run_dir, adapter, sha256_directory(adapter)

    runner.run_stage(
        "dpo_calibration",
        input_checkpoint_sha256=sft_sha,
        action=lambda child_id: dpo(child_id, calibration=True),
    )
    dpo_dir, dpo_adapter, dpo_sha = runner.run_stage(
        "dpo",
        input_checkpoint_sha256=sft_sha,
        action=lambda child_id: dpo(child_id, calibration=False),
    )
    if dpo_dir is None or dpo_adapter is None or dpo_sha is None:
        raise RuntimeError("DPO did not produce an adapter")
    dpo_run_id = dpo_dir.name
    dpo_eval_dir, _, _ = runner.run_stage(
        "dpo_eval",
        input_checkpoint_sha256=dpo_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="dpo",
            adapter=dpo_adapter,
            adapter_name="dpo",
            limit=None,
            parents=[dpo_run_id],
        ),
    )
    if dpo_eval_dir is None:
        raise RuntimeError("DPO evaluation did not produce a child run")

    grpo_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_grpo",
    )
    grpo_calibration_result = _resolved(
        repository,
        hardware=recipe.hardware,
        model=recipe.model,
        experiment="math_grpo",
        overrides=(
            "rollout.group_size=2",
            "optimization.iterations=1",
            "optimization.minibatch_completions=2",
            "optimization.gradient_accumulation_steps=1",
        ),
    )

    def grpo(child_id: str, *, calibration: bool) -> StageResult:
        context = execute_grpo_run(
            grpo_calibration_result if calibration else grpo_result,
            tasks_path=tasks_path,
            dpo_adapter_path=dpo_adapter,
            artifacts_root=runs_root,
            run_id=child_id,
            local_files_only=local_files_only,
            device=device,
            iteration_limit=1 if calibration else None,
        )
        _attach_to_pipeline(context, pipeline_run_id=selected_run_id, parent_run_ids=[dpo_run_id])
        adapter = context.run_dir / "adapter" / "grpo"
        return context.run_dir, adapter, sha256_directory(adapter)

    runner.run_stage(
        "grpo_calibration",
        input_checkpoint_sha256=dpo_sha,
        action=lambda child_id: grpo(child_id, calibration=True),
    )
    grpo_dir, grpo_adapter, grpo_sha = runner.run_stage(
        "grpo",
        input_checkpoint_sha256=dpo_sha,
        action=lambda child_id: grpo(child_id, calibration=False),
    )
    if grpo_dir is None or grpo_adapter is None or grpo_sha is None:
        raise RuntimeError("GRPO did not produce an adapter")
    grpo_run_id = grpo_dir.name
    grpo_eval_dir, _, _ = runner.run_stage(
        "grpo_eval",
        input_checkpoint_sha256=grpo_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="grpo",
            adapter=grpo_adapter,
            adapter_name="grpo",
            limit=None,
            parents=[grpo_run_id],
        ),
    )
    repeat_dir, _, _ = runner.run_stage(
        "grpo_eval_repeat",
        input_checkpoint_sha256=grpo_sha,
        action=lambda child_id: evaluate(
            child_id,
            checkpoint_id="grpo",
            adapter=grpo_adapter,
            adapter_name="grpo",
            limit=None,
            parents=[grpo_run_id],
        ),
    )
    if grpo_eval_dir is None or repeat_dir is None:
        raise RuntimeError("GRPO evaluations did not produce child runs")

    evaluations = {
        "base": base_eval_dir,
        "sft": sft_eval_dir,
        "dpo": dpo_eval_dir,
        "grpo": grpo_eval_dir,
    }
    training_runs = {"sft": sft_dir, "dpo": dpo_dir, "grpo": grpo_dir}
    checkpoint_hashes = {"base": base_sha, "sft": sft_sha, "dpo": dpo_sha, "grpo": grpo_sha}

    def report(_child_id: str) -> StageResult:
        hashes = build_pipeline_report(
            pipeline_dir,
            evaluations=evaluations,
            training_runs=training_runs,
            checkpoint_hashes=checkpoint_hashes,
            repeat_evaluation=repeat_dir,
        )
        return None, pipeline_dir / "comparison.json", hashes["comparison.json"]

    runner.run_stage("report", input_checkpoint_sha256=grpo_sha, action=report)
    phase_peaks: dict[str, int] = {}
    for stage in runner.manifest.stages:
        attempt = stage.attempts[-1]
        if attempt.run_directory is None:
            continue
        child_dir = runner.directory / attempt.run_directory
        summary_path = child_dir / "summary.json"
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        child_manifest = _read_json(child_dir / "run_manifest.json")
        evaluation_record = child_manifest.get("evaluation", {})
        evaluation_peak = (
            evaluation_record.get("peak_reserved_bytes", 0)
            if isinstance(evaluation_record, dict)
            else 0
        )
        peak_value = summary.get("peak_reserved_bytes", evaluation_peak)
        if not isinstance(peak_value, int) or isinstance(peak_value, bool):
            raise ValueError(f"stage {stage.id!r} has a non-integer peak-memory measurement")
        phase_peaks[stage.id] = peak_value
    runner.manifest.phase_peak_reserved_bytes = phase_peaks
    runner.manifest.final_checkpoint_sha256 = grpo_sha
    runner.manifest.comparison_artifacts = {
        name: sha256_file(pipeline_dir / name)
        for name in ("comparison.json", "report.md", "report.html")
    }
    runner.manifest.total_wall_seconds = sum(
        attempt.wall_seconds for stage in runner.manifest.stages for attempt in stage.attempts
    )
    runner.manifest.status = "completed"
    runner.manifest.finished_at = utc_now()
    runner.write()
    return pipeline_dir
