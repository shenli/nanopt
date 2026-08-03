"""In-process host backend for trusted unit tests; never secure isolation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from nanopt.agent.sandbox.base import SandboxExecution, SandboxLimits
from nanopt.agent.sandbox.process import run_bounded_process


class FakeSandboxBackend:
    """Run trusted fixtures locally with sanitized environment and bounded capture."""

    name: Literal["fake"] = "fake"

    def run(self, command: list[str], workspace: Path, limits: SandboxLimits) -> SandboxExecution:
        if not command:
            raise ValueError("trusted test command must not be empty")
        # Task cards say ``python`` for portability; use the current locked interpreter in tests.
        arguments = [sys.executable if command[0] == "python" else command[0], *command[1:]]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
        }
        return run_bounded_process(
            arguments,
            cwd=workspace,
            timeout_seconds=limits.timeout_seconds,
            output_bytes=limits.output_bytes,
            environment=environment,
            backend_details={"secure_isolation": False, "backend": self.name},
        )
