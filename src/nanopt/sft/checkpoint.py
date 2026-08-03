"""Adapter-only SFT checkpoints saved exclusively at optimizer boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, Field

from nanopt.models.adapters import save_lora_adapter
from nanopt.runtime.artifacts import sha256_bytes, sha256_file, write_json
from nanopt.sft.trainer import SftTrainingState


class SftCheckpointMetadata(BaseModel):
    """Small JSON document that identifies every checkpoint component."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    adapter_name: str
    adapter_path: str
    adapter_sha256: str
    optimizer_path: str
    optimizer_sha256: str
    optimizer_step: int = Field(ge=0)
    total_optimizer_steps: int = Field(gt=0)


def sha256_directory(path: Path) -> str:
    """Hash relative names and contents of every regular file in a directory tree."""

    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    pieces = bytearray()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        pieces.extend(len(relative).to_bytes(8, "big"))
        pieces.extend(relative)
        pieces.extend(bytes.fromhex(sha256_file(item)))
    return sha256_bytes(bytes(pieces))


def save_sft_checkpoint(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    state: SftTrainingState,
    path: Path,
    *,
    adapter_name: str,
) -> SftCheckpointMetadata:
    """Save adapter, optimizer, and RNG state into one new checkpoint directory."""

    if state.optimizer_step <= 0:
        raise ValueError("SFT checkpoints require at least one completed optimizer step")
    path.mkdir(parents=True, exist_ok=False)
    adapter_dir = save_lora_adapter(model, path / "adapter", adapter_name=adapter_name)
    optimizer_path = path / "optimizer.pt"
    temporary = path / ".optimizer.pt.tmp"
    payload: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(payload, temporary)
    os.replace(temporary, optimizer_path)
    metadata = SftCheckpointMetadata(
        adapter_name=adapter_name,
        adapter_path=adapter_dir.relative_to(path).as_posix(),
        adapter_sha256=sha256_directory(adapter_dir),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=sha256_file(optimizer_path),
        optimizer_step=state.optimizer_step,
        total_optimizer_steps=state.total_optimizer_steps,
    )
    write_json(path / "checkpoint.json", metadata.model_dump(mode="json"))
    return metadata


def read_sft_checkpoint(path: Path) -> SftCheckpointMetadata:
    """Validate checkpoint metadata and hashes before model or optimizer restoration."""

    try:
        metadata = SftCheckpointMetadata.model_validate_json(
            (path / "checkpoint.json").read_text(encoding="utf-8"), strict=True
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid SFT checkpoint metadata under {path}: {exc}") from exc
    adapter_dir = path / metadata.adapter_path
    optimizer_path = path / metadata.optimizer_path
    if sha256_directory(adapter_dir) != metadata.adapter_sha256:
        raise ValueError("SFT adapter checkpoint hash does not match metadata")
    if sha256_file(optimizer_path) != metadata.optimizer_sha256:
        raise ValueError("SFT optimizer checkpoint hash does not match metadata")
    return metadata


def restore_sft_optimizer(
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    metadata: SftCheckpointMetadata,
) -> None:
    """Restore trusted local optimizer and RNG state after the adapter is attached."""

    payload = torch.load(
        checkpoint_dir / metadata.optimizer_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("optimizer"), dict):
        raise ValueError("SFT optimizer checkpoint payload is malformed")
    optimizer.load_state_dict(payload["optimizer"])
    cpu_rng_state = payload.get("cpu_rng_state")
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise ValueError("SFT checkpoint is missing CPU RNG state")
    torch.set_rng_state(cpu_rng_state)
    cuda_states = payload.get("cuda_rng_states")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
