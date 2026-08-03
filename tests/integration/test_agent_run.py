from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from nanopt.agent.run import execute_agent_run
from nanopt.config.resolver import resolve_config


def test_fake_agent_run_writes_reports_and_exact_replay(project_root: Path, tmp_path: Path) -> None:
    result = resolve_config(
        hardware_id="rtx_4070_ti_super_16gb",
        model_id="qwen3_0_6b_base",
        experiment_id="mini_swe_rollout",
        overrides=("environment.backend=fake",),
    )
    context = execute_agent_run(
        result,
        tasks_root=project_root / "tasks/mini_swe_v1",
        policy_kind="oracle",
        artifacts_root=tmp_path,
        run_id="agent-integration",
        adapter_path=None,
        adapter_name="grpo",
        local_files_only=True,
        device="cpu",
    )
    summary = json.loads((context.run_dir / "summary.json").read_text())
    assert summary["solved"] == summary["tasks"] == 5
    assert summary["environment_trains_model"] is False
    replay = json.loads((context.run_dir / "replay.json").read_text())
    assert all(value["exact_semantic_match"] for value in replay.values())
    manifest = json.loads((context.run_dir / "run_manifest.json").read_text())
    run_schema = json.loads((project_root / "specs/schemas/run_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(run_schema).validate(manifest)
    assert manifest["agent_environment"]["hidden_source_exposed"] is False
    assert manifest["agent_environment"]["environment_trains_model"] is False
