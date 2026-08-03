from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nanopt.agent.sandbox import DockerSandboxBackend, FakeSandboxBackend, SandboxLimits

DIGEST = "python:3.11-slim@sha256:" + "a" * 64


def test_docker_command_contains_every_security_boundary(tmp_path: Path) -> None:
    backend = DockerSandboxBackend(DIGEST)
    command = backend.command(["python", "-m", "unittest"], tmp_path, SandboxLimits(10, 128, 16))
    joined = " ".join(command)
    for required in (
        "--network none",
        "--user 65532:65532",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "--read-only",
        "--memory 128m",
        "--pids-limit 16",
    ):
        assert required in joined
    assert "--gpus" not in command
    assert "docker.sock" not in joined
    mount = command[command.index("--mount") + 1]
    assert mount.endswith("dst=/workspace")
    assert ",rw" not in mount


def test_docker_backend_requires_digest() -> None:
    with pytest.raises(ValueError, match="pinned"):
        DockerSandboxBackend("python:3.11-slim")


def test_fake_backend_bounds_output_and_timeout(tmp_path: Path) -> None:
    backend = FakeSandboxBackend()
    output = backend.run(
        ["python", "-c", "print('x' * 10000)"],
        tmp_path,
        SandboxLimits(5, 128, 16, output_bytes=100),
    )
    assert output.output_truncated is True
    assert len(output.output.encode()) <= 100

    timeout = backend.run(
        ["python", "-c", "import time; time.sleep(2)"],
        tmp_path,
        SandboxLimits(1, 128, 16),
    )
    assert timeout.status == "timeout"
    assert timeout.exit_code is None
    assert sys.executable
