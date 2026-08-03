"""Orchestrate one inspectable DPO run from a frozen SFT adapter."""

from __future__ import annotations

import html
import math
from pathlib import Path

import torch

from nanopt.config.models import DpoExperiment
from nanopt.config.resolver import ResolutionResult
from nanopt.data.preferences import PreferenceAudit, PreferencePair, read_preference_pairs
from nanopt.dpo.cache import (
    ReferenceCacheIdentity,
    build_reference_cache,
    reference_cache_parity_error,
)
from nanopt.dpo.data import PreferenceCollator, render_preference_pairs
from nanopt.dpo.records import DpoMetricRecord, DpoSummary
from nanopt.dpo.trainer import DpoBatchMetrics, DpoStepMetrics, evaluate_dpo, train_dpo
from nanopt.models.adapters import (
    clone_lora_adapter,
    load_lora_adapter,
    parameter_counts,
    save_lora_adapter,
)
from nanopt.models.loading import load_qwen3_base, qwen_chat_terminator_id
from nanopt.models.renderer import ChatRenderer
from nanopt.runtime.artifacts import append_jsonl, sha256_file, write_json, write_text
from nanopt.runtime.run_context import RunContext, create_run_context
from nanopt.sft.checkpoint import sha256_directory


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _read_audit(path: Path) -> PreferenceAudit:
    try:
        return PreferenceAudit.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid preference audit {path}: {exc}") from exc


def _split_pairs(
    pairs: list[PreferencePair], experiment: DpoExperiment
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    train = [pair for pair in pairs if pair.split == experiment.data.train_split]
    validation = [pair for pair in pairs if pair.split == experiment.data.validation_split]
    if not train or not validation:
        raise ValueError("DPO preferences need non-empty configured train and validation splits")
    return train, validation


def _metric_record(
    run_id: str,
    split: str,
    optimizer_step: int,
    metric: DpoBatchMetrics,
    *,
    learning_rate: float | None = None,
    gradient_norm: float | None = None,
    gradient_clipped: bool | None = None,
    peak_allocated_bytes: int = 0,
    peak_reserved_bytes: int = 0,
) -> DpoMetricRecord:
    return DpoMetricRecord(
        run_id=run_id,
        split="train" if split == "train" else "validation",
        optimizer_step=optimizer_step,
        dpo_loss=metric.loss,
        policy_chosen_logp=metric.policy_chosen_logp,
        policy_rejected_logp=metric.policy_rejected_logp,
        policy_margin=metric.policy_margin,
        reference_margin=metric.reference_margin,
        implicit_reward_margin=metric.implicit_reward_margin,
        preference_accuracy=metric.preference_accuracy,
        reward_accuracy=metric.reward_accuracy,
        pair_count=metric.pair_count,
        chosen_active_tokens=metric.chosen_active_tokens,
        rejected_active_tokens=metric.rejected_active_tokens,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        gradient_clipped=gradient_clipped,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
    )


def _step_as_batch(metric: DpoStepMetrics) -> DpoBatchMetrics:
    return DpoBatchMetrics(
        loss=metric.loss,
        policy_chosen_logp=metric.policy_chosen_logp,
        policy_rejected_logp=metric.policy_rejected_logp,
        policy_margin=metric.policy_margin,
        reference_margin=metric.reference_margin,
        implicit_reward_margin=metric.implicit_reward_margin,
        preference_accuracy=metric.preference_accuracy,
        reward_accuracy=metric.reward_accuracy,
        pair_count=metric.pair_count,
        chosen_active_tokens=metric.chosen_active_tokens,
        rejected_active_tokens=metric.rejected_active_tokens,
    )


def _write_report(run_dir: Path, summary: DpoSummary) -> None:
    policy_margin_row = (
        f"| Policy chosen margin | {summary.initial_validation_policy_margin:.6f} | "
        f"{summary.final_validation_policy_margin:.6f} |"
    )
    reward_accuracy_row = (
        f"| Implicit reward accuracy | {summary.initial_validation_reward_accuracy:.2%} | "
        f"{summary.final_validation_reward_accuracy:.2%} |"
    )
    markdown = f"""# NanoPT DPO Report

> Preference-stage evidence. Protected generation quality is measured by a separate evaluation.

## Lineage

- Run: `{summary.run_id}`
- Frozen SFT adapter: `{summary.sft_adapter_sha256}`
- DPO adapter: `{summary.dpo_adapter_sha256}`
- Reference cache: `{summary.reference_cache_sha256}`
- Cache/live maximum absolute error: {summary.reference_cache_parity_max_abs_error:.3g}
- Representative run: {str(summary.representative).lower()}

## Held-out preferences

| Metric | SFT policy copy | Final DPO policy |
| --- | ---: | ---: |
| DPO loss | {summary.initial_validation_loss:.6f} | {summary.final_validation_loss:.6f} |
{policy_margin_row}
{reward_accuracy_row}

The policy began as an exact copy of the frozen SFT adapter. Sequence log probabilities are FP32
masked sums over completion tokens. The cache identity binds the model, adapter, renderer, dataset,
length policy, EOS policy, and reduction convention.
"""
    write_text(run_dir / "report.md", markdown)
    write_text(
        run_dir / "report.html",
        "<!doctype html><html><head><meta charset='utf-8'><title>NanoPT DPO Report</title>"
        "</head><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
    )


def execute_dpo_run(
    result: ResolutionResult,
    *,
    preferences_path: Path,
    sft_adapter_path: Path,
    artifacts_root: Path,
    run_id: str | None,
    local_files_only: bool,
    device: str,
    pair_limit: int | None = None,
) -> RunContext:
    """Cache SFT scores, clone its adapter, optimize DPO, validate, and report."""

    experiment = result.config.experiment
    if not isinstance(experiment, DpoExperiment):
        raise ValueError("DPO execution requires a dpo experiment profile")
    if experiment.data.sequence_logprob_reduction != "sum":
        raise ValueError("NanoPT M5 implements the specified sum sequence reduction")
    pairs = read_preference_pairs(preferences_path)
    audit_path = preferences_path.with_name("preference_audit.json")
    audit = _read_audit(audit_path)
    train_pairs, validation_pairs = _split_pairs(pairs, experiment)
    representative = pair_limit is None
    if pair_limit is not None:
        if pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        train_pairs = train_pairs[:pair_limit]
    selected_pairs = [*train_pairs, *validation_pairs]
    selected_ids = {pair.pair_id for pair in selected_pairs}
    if representative and (
        len(selected_pairs) != audit.pair_count or len(selected_ids) != len(pairs)
    ):
        raise ValueError("representative DPO run must cache the complete preference dataset")

    context = create_run_context(result, artifacts_root=artifacts_root, run_id=run_id)
    try:
        context.set_status("running")
        torch.manual_seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment.seed)
        loaded = load_qwen3_base(result.config.model, local_files_only=local_files_only)
        policy = load_lora_adapter(
            loaded.model,
            sft_adapter_path,
            adapter_name=experiment.reference.checkpoint_stage,
            trainable=False,
        )
        selected_device = _select_device(device)
        policy.to(selected_device)
        renderer = ChatRenderer(
            loaded.tokenizer,
            enable_thinking=result.config.model.renderer.enable_thinking,
            terminal_token_id=qwen_chat_terminator_id(loaded.tokenizer),
        )
        rendered = render_preference_pairs(selected_pairs, renderer)
        rendered_by_id = {example.pair.pair_id: example for example in rendered}
        train_examples = [rendered_by_id[pair.pair_id] for pair in train_pairs]
        validation_examples = [rendered_by_id[pair.pair_id] for pair in validation_pairs]
        plain_collator = PreferenceCollator(
            pad_token_id=int(loaded.tokenizer.pad_token_id),
            max_prompt_length=experiment.data.max_prompt_length,
            max_completion_length=experiment.data.max_completion_length,
        )
        sft_sha = sha256_directory(sft_adapter_path)
        identity = ReferenceCacheIdentity(
            model_id=result.config.model.source.model_id,
            model_revision=loaded.model_revision,
            tokenizer_revision=loaded.tokenizer_revision,
            sft_adapter_sha256=sft_sha,
            chat_template_sha256=renderer.chat_template_sha256,
            preference_dataset_fingerprint=audit.dataset_fingerprint,
            max_prompt_length=experiment.data.max_prompt_length,
            max_completion_length=experiment.data.max_completion_length,
        )
        cache_manifest, cache_values = build_reference_cache(
            policy,
            rendered,
            plain_collator,
            identity=identity,
            output_dir=context.run_dir / "reference_cache",
            micro_batch_size=experiment.training.pair_micro_batch_size,
            device=selected_device,
        )
        parity_error = reference_cache_parity_error(
            policy,
            rendered,
            plain_collator,
            cache_values,
            sample_size=experiment.reference.cache_validation_sample_size,
            micro_batch_size=experiment.training.pair_micro_batch_size,
            device=selected_device,
        )
        if parity_error > 1e-5:
            raise ValueError(f"reference cache parity error {parity_error:.6g} exceeds 1e-5")

        clone_lora_adapter(
            policy,
            source_name=experiment.reference.checkpoint_stage,
            target_name=experiment.policy.adapter_name,
            trainable=True,
        )
        if experiment.training.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            policy.enable_input_require_grads()
        policy.config.use_cache = False
        collator = PreferenceCollator(
            pad_token_id=int(loaded.tokenizer.pad_token_id),
            max_prompt_length=experiment.data.max_prompt_length,
            max_completion_length=experiment.data.max_completion_length,
            reference_values=cache_values,
        )
        initial = evaluate_dpo(
            policy, validation_examples, collator, experiment.training, device=selected_device
        )
        run_id_value = str(context.manifest["run_id"])
        append_jsonl(
            context.run_dir / "metrics.jsonl",
            _metric_record(run_id_value, "validation", 0, initial).model_dump(mode="json"),
        )

        def on_step(metric: DpoStepMetrics) -> None:
            append_jsonl(
                context.run_dir / "metrics.jsonl",
                _metric_record(
                    run_id_value,
                    "train",
                    metric.optimizer_step,
                    _step_as_batch(metric),
                    learning_rate=metric.learning_rate,
                    gradient_norm=metric.gradient_norm,
                    gradient_clipped=metric.gradient_clipped,
                    peak_allocated_bytes=metric.peak_allocated_bytes,
                    peak_reserved_bytes=metric.peak_reserved_bytes,
                ).model_dump(mode="json"),
            )

        training = train_dpo(
            policy,
            train_examples,
            collator,
            experiment.training,
            seed=experiment.seed,
            device=selected_device,
            on_step=on_step,
        )
        final = evaluate_dpo(
            policy, validation_examples, collator, experiment.training, device=selected_device
        )
        append_jsonl(
            context.run_dir / "metrics.jsonl",
            _metric_record(run_id_value, "validation", training.optimizer_steps, final).model_dump(
                mode="json"
            ),
        )
        breakdown: dict[str, dict[str, float | int]] = {}
        for rejection_type in ("wrong_answer", "malformed_answer", "trailing_content"):
            subset = [
                example
                for example in validation_examples
                if example.pair.rejection_type == rejection_type
            ]
            if subset:
                values = evaluate_dpo(
                    policy, subset, collator, experiment.training, device=selected_device
                )
                breakdown[rejection_type] = {
                    "pair_count": values.pair_count,
                    "loss": values.loss,
                    "policy_margin": values.policy_margin,
                    "reward_accuracy": values.reward_accuracy,
                }
        write_json(context.run_dir / "preference_breakdown.json", breakdown)
        adapter_dir = save_lora_adapter(
            policy,
            context.run_dir / "adapter",
            adapter_name=experiment.policy.adapter_name,
        )
        dpo_sha = sha256_directory(adapter_dir)
        peaks = training.metrics[-1]
        summary = DpoSummary(
            run_id=run_id_value,
            optimizer_steps=training.optimizer_steps,
            train_pairs=len(train_pairs),
            validation_pairs=len(validation_pairs),
            initial_validation_loss=initial.loss,
            final_validation_loss=final.loss,
            initial_validation_policy_margin=initial.policy_margin,
            final_validation_policy_margin=final.policy_margin,
            initial_validation_reward_accuracy=initial.reward_accuracy,
            final_validation_reward_accuracy=final.reward_accuracy,
            reference_cache_sha256=cache_manifest.cache_sha256,
            reference_cache_parity_max_abs_error=parity_error,
            sft_adapter_sha256=sft_sha,
            dpo_adapter_sha256=dpo_sha,
            peak_allocated_bytes=peaks.peak_allocated_bytes,
            peak_reserved_bytes=peaks.peak_reserved_bytes,
            representative=representative,
        )
        write_json(context.run_dir / "summary.json", summary.model_dump(mode="json"))
        _write_report(context.run_dir, summary)
        counts = parameter_counts(policy)
        context.manifest["model"].update(
            {
                "resolved_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "chat_template_sha256": renderer.chat_template_sha256,
                "trainable_parameter_count": counts.trainable,
                "parent_adapter_sha256": sft_sha,
                "adapter_name": experiment.policy.adapter_name,
                "adapter_sha256": dpo_sha,
            }
        )
        context.manifest["data"]["fingerprints"].update(
            {
                "preference_file_sha256": sha256_file(preferences_path),
                "preference_audit_sha256": sha256_file(audit_path),
                "preference_dataset": audit.dataset_fingerprint,
                "source_dataset": audit.source_dataset_fingerprint,
            }
        )
        context.manifest["training"] = {
            "device": selected_device.type,
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "optimizer_steps": training.optimizer_steps,
            "representative": representative,
            "initial_policy_is_exact_sft_copy": math.isclose(
                initial.loss, math.log(2), rel_tol=0, abs_tol=1e-5
            ),
        }
        context.manifest["artifacts"] = [
            {"path": name, "sha256": sha256_file(context.run_dir / name)}
            for name in (
                "summary.json",
                "metrics.jsonl",
                "preference_breakdown.json",
                "report.md",
                "report.html",
            )
        ]
        context.set_status("completed")
        return context
    except Exception as exc:
        context.set_status(
            "failed", failure={"type": type(exc).__name__, "message": str(exc), "phase": "dpo"}
        )
        raise
