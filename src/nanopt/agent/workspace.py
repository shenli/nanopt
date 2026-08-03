"""Bounded workspace tools with one audited path and patch-validation boundary."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nanopt.agent.records import AgentTaskCard, ToolResult
from nanopt.runtime.artifacts import sha256_bytes, sha256_file
from nanopt.sft.checkpoint import sha256_directory

CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


class WorkspacePolicyError(ValueError):
    """An untrusted path or patch violated the workspace policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def workspace_sha256(root: Path) -> str:
    """Hash the complete visible workspace and refuse ambiguous symbolic links."""

    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise WorkspacePolicyError(
                "symlink_forbidden", f"workspace contains symbolic link {entry.relative_to(root)}"
            )
    return sha256_directory(root)


@dataclass(frozen=True)
class _FilePatch:
    path: str
    hunks: tuple[tuple[int, tuple[str, ...]], ...]


class SafeWorkspace:
    """Implement model-visible file operations without accepting arbitrary commands."""

    def __init__(
        self,
        root: Path,
        card: AgentTaskCard,
        *,
        maximum_output_bytes: int = 32768,
        maximum_read_bytes: int = 32768,
        maximum_results: int = 200,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.card = card
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_read_bytes = maximum_read_bytes
        self.maximum_results = maximum_results

    @staticmethod
    def _clean_relative(value: str, *, allow_dot: bool = True) -> PurePosixPath:
        if not value or CONTROL_CHARACTERS.search(value) or "\\" in value:
            raise WorkspacePolicyError("invalid_path", "path contains invalid characters")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise WorkspacePolicyError("path_traversal", "path must remain workspace-relative")
        normalized = PurePosixPath(*[part for part in path.parts if part not in {"", "."}])
        if str(normalized) == "." and not allow_dot:
            raise WorkspacePolicyError("invalid_path", "a file path is required")
        return normalized

    def resolve(self, value: str, *, must_exist: bool, allow_directory: bool = True) -> Path:
        relative = self._clean_relative(value, allow_dot=allow_directory)
        candidate = self.root.joinpath(*relative.parts)
        current = self.root
        for component in relative.parts:
            current = current / component
            if current.exists() or current.is_symlink():
                mode = current.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise WorkspacePolicyError(
                        "symlink_escape", f"symbolic links are forbidden: {relative}"
                    )
        if must_exist and not candidate.exists():
            raise WorkspacePolicyError("not_found", f"workspace path does not exist: {relative}")
        if candidate.exists() and not allow_directory and not candidate.is_file():
            raise WorkspacePolicyError("not_file", f"workspace path is not a file: {relative}")
        resolved_boundary = (
            candidate.resolve(strict=True)
            if candidate.exists()
            else candidate.parent.resolve(strict=True)
        )
        if not resolved_boundary.is_relative_to(self.root):
            raise WorkspacePolicyError("path_traversal", "resolved path escapes workspace")
        return candidate

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _is_protected(self, relative: str) -> bool:
        return any(fnmatch.fnmatchcase(relative, pattern) for pattern in self.card.protected_globs)

    def _is_editable(self, relative: str) -> bool:
        return any(fnmatch.fnmatchcase(relative, pattern) for pattern in self.card.editable_globs)

    def list_files(self, path: str, max_depth: int) -> ToolResult:
        base = self.resolve(path, must_exist=True)
        if not base.is_dir():
            raise WorkspacePolicyError("not_directory", "list_files requires a directory")
        values: list[dict[str, str | int]] = []
        for entry in sorted(base.rglob("*")):
            relative_to_base = entry.relative_to(base)
            if len(relative_to_base.parts) > max_depth:
                continue
            if entry.is_symlink():
                kind = "symlink_forbidden"
            elif entry.is_dir():
                kind = "directory"
            elif entry.is_file():
                kind = "file"
            else:
                kind = "other"
            values.append({"path": self._relative(entry), "kind": kind})
            if len(values) >= self.maximum_results:
                break
        return ToolResult(
            status="ok",
            code="listed",
            message=f"listed {len(values)} entries",
            data={"entries": values},
            truncated=len(values) >= self.maximum_results,
        )

    def read_file(self, path: str, start_line: int, end_line: int) -> ToolResult:
        if end_line < start_line or end_line - start_line + 1 > 500:
            raise WorkspacePolicyError("invalid_range", "line range must contain 1 to 500 lines")
        target = self.resolve(path, must_exist=True, allow_directory=False)
        raw = target.read_bytes()
        if len(raw) > self.maximum_read_bytes:
            raise WorkspacePolicyError("file_too_large", "file exceeds the read byte limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspacePolicyError("binary_file", "file is not UTF-8 text") from exc
        lines = text.splitlines()
        selected = [
            {"line": index, "text": lines[index - 1]}
            for index in range(start_line, min(end_line, len(lines)) + 1)
        ]
        return ToolResult(
            status="ok",
            code="read",
            message=f"read {len(selected)} lines",
            data={"path": self._relative(target), "lines": selected, "total_lines": len(lines)},
        )

    def search(self, query: str, path: str, glob: str) -> ToolResult:
        if CONTROL_CHARACTERS.search(query):
            raise WorkspacePolicyError("invalid_query", "search query contains control characters")
        if not glob or ".." in PurePosixPath(glob).parts or CONTROL_CHARACTERS.search(glob):
            raise WorkspacePolicyError("invalid_glob", "search glob is invalid")
        base = self.resolve(path, must_exist=True)
        if not base.is_dir():
            raise WorkspacePolicyError("not_directory", "search requires a directory")
        matches: list[dict[str, str | int]] = []
        for candidate in sorted(base.rglob(glob)):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if candidate.stat().st_size > self.maximum_read_bytes:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        {"path": self._relative(candidate), "line": line_number, "text": line}
                    )
                    if len(matches) >= self.maximum_results:
                        return ToolResult(
                            status="ok",
                            code="searched",
                            message=f"found at least {len(matches)} matches",
                            data={"matches": matches},
                            truncated=True,
                        )
        return ToolResult(
            status="ok",
            code="searched",
            message=f"found {len(matches)} matches",
            data={"matches": matches},
        )

    @staticmethod
    def _patch_path(line: str, prefix: str) -> str:
        if not line.startswith(prefix):
            raise WorkspacePolicyError("invalid_patch", f"expected {prefix.strip()} header")
        value = line[len(prefix) :].split("\t", 1)[0].strip()
        if value == "/dev/null":
            raise WorkspacePolicyError("invalid_patch", "file creation/deletion is not supported")
        for marker in ("a/", "b/"):
            if value.startswith(marker):
                value = value[len(marker) :]
                break
        return value

    def _parse_patch(self, patch: str) -> list[_FilePatch]:
        lines = patch.splitlines(keepends=True)
        if not lines or len(patch.encode("utf-8")) > 65536:
            raise WorkspacePolicyError("invalid_patch", "patch is empty or exceeds 64 KiB")
        parsed: list[_FilePatch] = []
        index = 0
        while index < len(lines):
            if lines[index].startswith("diff --git "):
                index += 1
                continue
            old_path = self._patch_path(lines[index].rstrip("\n"), "--- ")
            index += 1
            if index >= len(lines):
                raise WorkspacePolicyError("invalid_patch", "patch is missing new-file header")
            new_path = self._patch_path(lines[index].rstrip("\n"), "+++ ")
            index += 1
            if old_path != new_path:
                raise WorkspacePolicyError("invalid_patch", "renames are not supported")
            hunks: list[tuple[int, tuple[str, ...]]] = []
            while index < len(lines) and not lines[index].startswith("--- "):
                header = HUNK_HEADER.match(lines[index].rstrip("\n"))
                if header is None:
                    raise WorkspacePolicyError("invalid_patch", "malformed unified-diff hunk")
                old_start = int(header.group(1))
                index += 1
                body: list[str] = []
                while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                    line = lines[index]
                    if line.startswith("\\ No newline at end of file"):
                        index += 1
                        continue
                    if not line.startswith((" ", "+", "-")):
                        raise WorkspacePolicyError("invalid_patch", "invalid hunk body prefix")
                    body.append(line)
                    index += 1
                if not body:
                    raise WorkspacePolicyError("invalid_patch", "empty patch hunk")
                hunks.append((old_start, tuple(body)))
            parsed.append(_FilePatch(old_path, tuple(hunks)))
        if not parsed or len({item.path for item in parsed}) != len(parsed):
            raise WorkspacePolicyError("invalid_patch", "patch targets are empty or duplicated")
        return parsed

    @staticmethod
    def _apply_hunks(original: str, file_patch: _FilePatch) -> str:
        source = original.splitlines(keepends=True)
        output: list[str] = []
        cursor = 0
        for old_start, body in file_patch.hunks:
            start = old_start - 1
            if start < cursor or start > len(source):
                raise WorkspacePolicyError("patch_context_mismatch", "hunk position is invalid")
            output.extend(source[cursor:start])
            cursor = start
            for line in body:
                prefix, content = line[0], line[1:]
                if prefix in {" ", "-"}:
                    if cursor >= len(source) or source[cursor] != content:
                        raise WorkspacePolicyError(
                            "patch_context_mismatch", "patch context does not match workspace"
                        )
                    if prefix == " ":
                        output.append(source[cursor])
                    cursor += 1
                else:
                    output.append(content)
        output.extend(source[cursor:])
        return "".join(output)

    def apply_patch(self, patch: str) -> ToolResult:
        parsed = self._parse_patch(patch)
        updates: list[tuple[Path, bytes, bytes]] = []
        for file_patch in parsed:
            target = self.resolve(file_patch.path, must_exist=True, allow_directory=False)
            relative = self._relative(target)
            if self._is_protected(relative):
                raise WorkspacePolicyError(
                    "protected_path", f"cannot modify protected path {relative}"
                )
            if not self._is_editable(relative):
                raise WorkspacePolicyError("path_not_editable", f"path is not editable: {relative}")
            raw = target.read_bytes()
            try:
                original = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspacePolicyError("binary_patch", "cannot patch a binary file") from exc
            replacement = self._apply_hunks(original, file_patch).encode("utf-8")
            updates.append((target, raw, replacement))

        temporary_paths: list[Path] = []
        replaced: list[tuple[Path, bytes]] = []
        try:
            for target, _old, replacement in updates:
                original_mode = stat.S_IMODE(target.stat().st_mode)
                descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
                temporary = Path(name)
                temporary_paths.append(temporary)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(replacement)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(original_mode)
            for (target, old, _replacement), temporary in zip(
                updates, temporary_paths, strict=True
            ):
                os.replace(temporary, target)
                replaced.append((target, old))
        except OSError as exc:
            for target, old in reversed(replaced):
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{target.name}.rollback.", dir=target.parent
                )
                rollback = Path(name)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(old)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(rollback, target)
            raise WorkspacePolicyError("patch_write_failed", f"atomic patch failed: {exc}") from exc
        finally:
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()
        changed = [self._relative(target) for target, _old, _new in updates]
        return ToolResult(
            status="ok",
            code="patched",
            message=f"patched {len(changed)} files",
            data={"paths": changed, "workspace_sha256": workspace_sha256(self.root)},
        )


def file_identity(path: Path) -> str:
    """Hash a path and content for final-patch lineage."""

    relative = path.name.encode("utf-8")
    return sha256_bytes(relative + bytes.fromhex(sha256_file(path)))
