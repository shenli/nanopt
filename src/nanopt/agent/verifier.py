"""Public and hidden verification with protected-file and workspace isolation checks."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from nanopt.agent.records import AgentTaskCard, AgentVerification, TestSummary
from nanopt.agent.sandbox.base import SandboxBackend, SandboxExecution, SandboxLimits
from nanopt.agent.tasks import LoadedAgentTask
from nanopt.agent.workspace import WorkspacePolicyError, workspace_sha256
from nanopt.runtime.artifacts import sha256_file

FAILURE_COUNT = re.compile(r"(?:failures|errors)=(\d+)")


def _make_container_readable(root: Path) -> None:
    root.chmod(0o777)
    for entry in root.rglob("*"):
        if entry.is_dir():
            entry.chmod(0o777)
        elif entry.is_file():
            entry.chmod(0o666)


def _summary(
    execution: SandboxExecution,
    *,
    total: int,
    workspace_hash: str,
    expose_output: bool,
) -> TestSummary:
    if execution.status == "timeout":
        status: Literal["passed", "failed", "timeout", "sandbox_error"] = "timeout"
        passed = 0
    elif execution.status == "sandbox_error":
        status = "sandbox_error"
        passed = 0
    elif execution.exit_code == 0:
        status = "passed"
        passed = total
    else:
        status = "failed"
        failed = sum(int(value) for value in FAILURE_COUNT.findall(execution.output))
        passed = max(0, total - max(1, failed))
    return TestSummary(
        status=status,
        passed=passed,
        total=total,
        exit_code=execution.exit_code,
        duration_seconds=execution.duration_seconds,
        output=execution.output if expose_output else None,
        output_truncated=execution.output_truncated,
        workspace_sha256=workspace_hash,
    )


def protected_file_hashes(root: Path, card: AgentTaskCard) -> dict[str, str]:
    """Capture public tests/metadata that model-visible tools must never alter."""

    import fnmatch

    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and any(
            fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern)
            for pattern in card.protected_globs
        )
    }


def protected_files_unchanged(workspace: Path, expected: dict[str, str]) -> tuple[bool, list[str]]:
    changed: list[str] = []
    for relative, digest in expected.items():
        path = workspace / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            changed.append(relative)
    return not changed, changed


class HiddenVerifier:
    """Run public and hidden checks in separate copies of the submitted workspace."""

    def __init__(self, backend: SandboxBackend, limits: SandboxLimits) -> None:
        self.backend = backend
        self.limits = limits

    def run_public(self, task: LoadedAgentTask, workspace: Path) -> TestSummary:
        """Expose bounded public output, but contain any test side effects in a copy."""

        visible_hash = workspace_sha256(workspace)
        with tempfile.TemporaryDirectory(prefix="nanopt-public-verifier-") as temporary:
            verifier_root = Path(temporary) / "workspace"
            shutil.copytree(workspace, verifier_root, symlinks=False)
            _make_container_readable(verifier_root)
            execution = self.backend.run(
                task.card.public_test_command,
                verifier_root,
                self.limits,
            )
        return _summary(
            execution,
            total=task.card.public_tests_total,
            workspace_hash=visible_hash,
            expose_output=True,
        )

    def run_hidden(self, task: LoadedAgentTask, workspace: Path) -> TestSummary:
        """Return counts only; never expose hidden paths, source, or output."""

        visible_hash = workspace_sha256(workspace)
        with tempfile.TemporaryDirectory(prefix="nanopt-hidden-verifier-") as temporary:
            verifier_root = Path(temporary) / "workspace"
            shutil.copytree(workspace, verifier_root, symlinks=False)
            hidden_destination = verifier_root / ".nanopt_hidden_tests"
            shutil.copytree(task.hidden_tests_dir, hidden_destination, symlinks=False)
            _make_container_readable(verifier_root)
            execution = self.backend.run(
                task.card.hidden_test_command,
                verifier_root,
                self.limits,
            )
        return _summary(
            execution,
            total=task.card.hidden_tests_total,
            workspace_hash=visible_hash,
            expose_output=False,
        )

    def verify(
        self,
        task: LoadedAgentTask,
        workspace: Path,
        *,
        expected_protected_files: dict[str, str],
        policy_violations: int,
    ) -> AgentVerification:
        try:
            unchanged, changed = protected_files_unchanged(workspace, expected_protected_files)
            if not unchanged:
                visible_hash = workspace_sha256(workspace)
                message = "protected files changed: " + ", ".join(changed)
                public = TestSummary(
                    status="failed",
                    passed=0,
                    total=task.card.public_tests_total,
                    exit_code=None,
                    duration_seconds=0,
                    output=message,
                    workspace_sha256=visible_hash,
                )
                hidden = TestSummary(
                    status="failed",
                    passed=0,
                    total=task.card.hidden_tests_total,
                    exit_code=None,
                    duration_seconds=0,
                    output=None,
                    workspace_sha256=visible_hash,
                )
            else:
                public = self.run_public(task, workspace)
                hidden = self.run_hidden(task, workspace)
        except WorkspacePolicyError as exc:
            fallback_hash = "0" * 64
            public = TestSummary(
                status="failed",
                passed=0,
                total=task.card.public_tests_total,
                exit_code=None,
                duration_seconds=0,
                output=str(exc),
                workspace_sha256=fallback_hash,
            )
            hidden = TestSummary(
                status="failed",
                passed=0,
                total=task.card.hidden_tests_total,
                exit_code=None,
                duration_seconds=0,
                output=None,
                workspace_sha256=fallback_hash,
            )
        raw_score = hidden.passed / hidden.total if public.status == "passed" else 0.0
        penalty = min(raw_score, policy_violations * 0.1)
        return AgentVerification(
            public=public,
            hidden=hidden,
            final_score=max(0.0, raw_score - penalty),
            policy_violation_penalty=penalty,
        )
