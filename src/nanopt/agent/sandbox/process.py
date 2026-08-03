"""Bounded subprocess capture used by sandbox implementations."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from nanopt.agent.sandbox.base import SandboxExecution

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def sanitize_output(raw: bytes, maximum_bytes: int) -> tuple[str, bool]:
    """Bound bytes and remove terminal controls that could forge surrounding logs."""

    truncated = len(raw) > maximum_bytes
    raw = raw[:maximum_bytes]
    text = raw.decode("utf-8", errors="replace")
    text = ANSI_ESCAPE.sub("", text)
    text = "".join(character for character in text if character in "\n\t" or ord(character) >= 32)
    return text, truncated


def run_bounded_process(
    arguments: list[str],
    *,
    cwd: Path | None,
    timeout_seconds: int,
    output_bytes: int,
    environment: dict[str, str] | None = None,
    backend_details: dict[str, str | int | float | bool | None] | None = None,
) -> SandboxExecution:
    """Run without a shell, kill the process group on timeout, and return bounded output."""

    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return SandboxExecution(
            status="sandbox_error",
            exit_code=None,
            output=str(exc),
            output_truncated=False,
            duration_seconds=time.perf_counter() - started,
            backend_details=backend_details or {},
        )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        status: Literal["completed", "timeout", "sandbox_error"] = "completed"
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        status = "timeout"
        exit_code = None
    bounded, truncated = sanitize_output(output, output_bytes)
    return SandboxExecution(
        status=status,
        exit_code=exit_code,
        output=bounded,
        output_truncated=truncated,
        duration_seconds=time.perf_counter() - started,
        backend_details=backend_details or {},
    )
