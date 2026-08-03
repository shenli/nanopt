"""Hardened local Docker backend for executing untrusted MiniSWE code."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from nanopt.agent.sandbox.base import SandboxExecution, SandboxLimits
from nanopt.agent.sandbox.process import run_bounded_process


class DockerSandboxBackend:
    """Run a trusted command with no network/GPU and a non-root, capability-free container."""

    name: Literal["docker"] = "docker"

    def __init__(self, image: str, *, executable: str = "docker") -> None:
        if "@sha256:" not in image:
            raise ValueError("validated Docker sandbox image must be pinned by sha256 digest")
        if not image.startswith("python:"):
            raise ValueError("M8 reference Docker image must be an official python image")
        self.image = image
        self.executable = executable

    def validate_available(self) -> dict[str, Any]:
        """Prove the daemon is reachable and the exact pinned image is already present."""

        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError("docker executable is unavailable")
        try:
            version = subprocess.run(
                [executable, "version", "--format", "{{json .Server.Version}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            inspected = subprocess.run(
                [executable, "image", "inspect", self.image],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            values = json.loads(inspected)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"pinned Docker sandbox image is unavailable: {exc}") from exc
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise RuntimeError("docker image inspection returned an unexpected payload")
        return {"server_version": json.loads(version), "image_id": values[0].get("Id")}

    def command(
        self, trusted_command: list[str], workspace: Path, limits: SandboxLimits
    ) -> list[str]:
        """Build the complete argv so security flags are unit-testable without running Docker."""

        if not trusted_command:
            raise ValueError("trusted test command must not be empty")
        resolved = workspace.resolve(strict=True)
        # A bind mount is writable by default. Docker 29 rejects a bare `rw` field because
        # `--mount` accepts either key/value pairs or the standalone `readonly` flag.
        mount = f"type=bind,src={resolved},dst=/workspace"
        return [
            self.executable,
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--memory",
            f"{limits.memory_mib}m",
            "--memory-swap",
            f"{limits.memory_mib}m",
            "--pids-limit",
            str(limits.pids),
            "--cpus",
            str(limits.cpus),
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONPATH=/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            self.image,
            *trusted_command,
        ]

    def run(self, command: list[str], workspace: Path, limits: SandboxLimits) -> SandboxExecution:
        arguments = self.command(command, workspace, limits)
        return run_bounded_process(
            arguments,
            cwd=None,
            timeout_seconds=limits.timeout_seconds,
            output_bytes=limits.output_bytes,
            backend_details={
                "secure_isolation": True,
                "backend": self.name,
                "image": self.image,
                "network": "none",
                "user": "65532:65532",
                "gpu_exposed": False,
                "capabilities_dropped": True,
                "no_new_privileges": True,
                "root_filesystem_read_only": True,
                "memory_mib": limits.memory_mib,
                "pids": limits.pids,
                "cpus": limits.cpus,
            },
        )
