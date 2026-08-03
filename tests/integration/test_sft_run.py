from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from nanopt.config.loader import ConfigRepository
from nanopt.config.resolver import resolve_config
from nanopt.data.arithmetic import ArithmeticGeneratorConfig, generate_tasks
from nanopt.data.schemas import ArithmeticSplitConfig
from nanopt.data.splits import SPLIT_ORDER, build_splits
from nanopt.models.adapters import parameter_counts
from nanopt.models.loading import LoadedModel
from nanopt.runtime.artifacts import append_jsonl, write_json
from nanopt.sft.run import execute_sft_run


class LocalTokenizer:
    """Deterministic chat template fixture with an exact assistant prefix."""

    chat_template = "fixture-template"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> list[int]:
        assert tokenize and not enable_thinking and return_tensors == "pt"
        role_ids = {"user": 10, "assistant": 11, "system": 12}
        ids = [1]
        for message in conversation:
            ids.extend(
                [role_ids[message["role"]], *[20 + ord(c) % 200 for c in message["content"]]]
            )
        if add_generation_prompt:
            ids.append(role_ids["assistant"])
        return ids


def _write_tasks(tmp_path: Path, project_root: Path) -> Path:
    generator = ArithmeticGeneratorConfig.model_validate(
        yaml.safe_load((project_root / "tasks/arithmetic/generator_config.yaml").read_text()),
        strict=True,
    )
    split = ArithmeticSplitConfig.model_validate(
        yaml.safe_load((project_root / "tasks/arithmetic/split_config.yaml").read_text()),
        strict=True,
    )
    splits, manifest = build_splits(
        generate_tasks(generator),
        counts=split.counts,
        seed=split.seed,
        generator_config=generator,
    )
    tasks_path = tmp_path / "data" / "tasks.jsonl"
    for name in SPLIT_ORDER:
        for task in splits[name]:
            append_jsonl(tasks_path, task.model_dump(mode="json", exclude_none=True))
    write_json(tasks_path.with_name("dataset_manifest.json"), manifest.model_dump(mode="json"))
    return tasks_path


def test_local_tiny_sft_run_writes_adapter_metrics_and_valid_manifest(
    monkeypatch: Any, tmp_path: Path, project_root: Path
) -> None:
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    loaded = LoadedModel(
        model=model,
        tokenizer=LocalTokenizer(),
        model_revision="fixture-model-revision",
        tokenizer_revision="fixture-tokenizer-revision",
        parameters=parameter_counts(model),
    )
    monkeypatch.setattr("nanopt.sft.run.load_qwen3_base", lambda *_args, **_kwargs: loaded)
    resolved = resolve_config(
        repository=ConfigRepository(project_root / "configs"),
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="math_sft",
        overrides=(
            "training.micro_batch_size=2",
            "training.gradient_accumulation_steps=1",
            "training.max_steps=1",
            "training.gradient_checkpointing=false",
        ),
    )

    context = execute_sft_run(
        resolved,
        tasks_path=_write_tasks(tmp_path, project_root),
        artifacts_root=tmp_path / "runs",
        run_id="tiny-sft",
        local_files_only=True,
        device="cpu",
        train_limit=2,
    )

    manifest = json.loads(context.manifest_path.read_text())
    schema = json.loads((project_root / "specs/schemas/run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["status"] == "completed"
    assert manifest["training"]["representative"] is False
    assert manifest["model"]["trainable_parameter_count"] > 0
    assert (
        context.run_dir / manifest["checkpoint"]["path"] / "adapter_model.safetensors"
    ).is_file()
    assert len((context.run_dir / "metrics.jsonl").read_text().splitlines()) == 3
    assert json.loads((context.run_dir / "summary.json").read_text())["optimizer_steps"] == 1
