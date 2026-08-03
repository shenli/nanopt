"""Minimal execution boundary shared by fake and Docker MiniSWE backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int
    memory_mib: int
    pids: int
    output_bytes: int = 32768
    cpus: float = 1.0

    def __post_init__(self) -> None:
        if min(self.timeout_seconds, self.memory_mib, self.pids, self.output_bytes) <= 0:
            raise ValueError("sandbox limits must be positive")
        if self.cpus <= 0:
            raise ValueError("sandbox CPU limit must be positive")


@dataclass(frozen=True)
class SandboxExecution:
    status: Literal["completed", "timeout", "sandbox_error"]
    exit_code: int | None
    output: str
    output_truncated: bool
    duration_seconds: float
    backend_details: dict[str, str | int | float | bool | None]


class SandboxBackend(Protocol):
    """Execute only a trusted task-defined command inside one supplied workspace."""

    @property
    def name(self) -> Literal["fake", "docker"]: ...

    def run(
        self,
        command: list[str],
        workspace: Path,
        limits: SandboxLimits,
    ) -> SandboxExecution: ...
