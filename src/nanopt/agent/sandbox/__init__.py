"""Sandbox backends for trusted tests and reference Docker execution."""

from nanopt.agent.sandbox.base import SandboxBackend, SandboxExecution, SandboxLimits
from nanopt.agent.sandbox.docker import DockerSandboxBackend
from nanopt.agent.sandbox.fake import FakeSandboxBackend

__all__ = [
    "DockerSandboxBackend",
    "FakeSandboxBackend",
    "SandboxBackend",
    "SandboxExecution",
    "SandboxLimits",
]
