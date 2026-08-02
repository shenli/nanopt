"""Read-only environment diagnosis with stable machine-readable output."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from nanopt.config.models import HardwareProfile


class DoctorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DependencyStatus(DoctorModel):
    required: bool
    installed: bool
    version: str | None


class GpuStatus(DoctorModel):
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: str
    bf16_supported: bool


class CudaStatus(DoctorModel):
    available: bool
    runtime_version: str | None
    driver_version: str | None
    device_count: int
    gpus: list[GpuStatus]


class DockerStatus(DoctorModel):
    executable_found: bool
    daemon_reachable: bool
    version: str | None


class ProfileMatch(DoctorModel):
    requested_id: str | None
    matched: bool
    support_status: str | None
    reasons: list[str]


class DoctorReport(DoctorModel):
    schema_version: Literal[1] = 1
    status: Literal["usable", "warning", "unusable", "profile_mismatch"]
    exit_code: Literal[0, 2, 3, 4]
    os: str
    architecture: str
    python_version: str
    pytorch_version: str | None
    dependencies: dict[str, DependencyStatus]
    cuda: CudaStatus
    tf32_available: bool
    huggingface_cache: str
    docker: DockerStatus
    profile: ProfileMatch
    messages: list[str]


REQUIRED_DEPENDENCIES = ("torch", "transformers", "peft", "pydantic", "yaml", "typer")
OPTIONAL_DEPENDENCIES = ("mkdocs", "trl", "bitsandbytes")


def _distribution_name(module_name: str) -> str:
    return {"yaml": "PyYAML"}.get(module_name, module_name)


def _dependencies() -> dict[str, DependencyStatus]:
    result: dict[str, DependencyStatus] = {}
    for module_name in (*REQUIRED_DEPENDENCIES, *OPTIONAL_DEPENDENCIES):
        distribution = _distribution_name(module_name)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[module_name] = DependencyStatus(
            required=module_name in REQUIRED_DEPENDENCIES,
            installed=version is not None,
            version=version,
        )
    return result


def _driver_version(torch_module: Any) -> str | None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None
    try:
        value = cuda.driver_version()
    except (AttributeError, RuntimeError):
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.splitlines()[0].strip() or None
    return str(value)


def _cuda_status(torch_module: Any | None) -> CudaStatus:
    if torch_module is None:
        return CudaStatus(
            available=False,
            runtime_version=None,
            driver_version=None,
            device_count=0,
            gpus=[],
        )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
        return CudaStatus(
            available=False,
            runtime_version=runtime,
            driver_version=_driver_version(torch_module),
            device_count=0,
            gpus=[],
        )

    count = int(cuda.device_count())
    gpus: list[GpuStatus] = []
    for index in range(count):
        properties = cuda.get_device_properties(index)
        try:
            free_bytes, total_bytes = cuda.mem_get_info(index)
        except (AttributeError, RuntimeError):
            total_bytes = int(properties.total_memory)
            free_bytes = 0
        major, minor = cuda.get_device_capability(index)
        try:
            bf16_supported = bool(cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            bf16_supported = bool(cuda.is_bf16_supported())
        gpus.append(
            GpuStatus(
                index=index,
                name=str(properties.name),
                total_vram_bytes=int(total_bytes),
                free_vram_bytes=int(free_bytes),
                compute_capability=f"{major}.{minor}",
                bf16_supported=bf16_supported,
            )
        )
    return CudaStatus(
        available=True,
        runtime_version=getattr(getattr(torch_module, "version", None), "cuda", None),
        driver_version=_driver_version(torch_module),
        device_count=count,
        gpus=gpus,
    )


def _docker_status() -> DockerStatus:
    executable = shutil.which("docker")
    if executable is None:
        return DockerStatus(executable_found=False, daemon_reachable=False, version=None)
    try:
        completed = subprocess.run(
            [executable, "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return DockerStatus(executable_found=True, daemon_reachable=False, version=None)
    return DockerStatus(
        executable_found=True,
        daemon_reachable=True,
        version=completed.stdout.strip() or None,
    )


def _profile_match(profile: HardwareProfile | None, cuda: CudaStatus) -> ProfileMatch:
    if profile is None:
        return ProfileMatch(requested_id=None, matched=False, support_status=None, reasons=[])
    reasons: list[str] = []
    actual_os = platform.system().lower()
    actual_arch = platform.machine().lower()
    expected_arch = profile.platform.architecture.lower()
    arch_aliases = {"amd64": "x86_64", "aarch64": "arm64"}
    if actual_os != profile.platform.os.lower():
        reasons.append(f"OS is {actual_os}, expected {profile.platform.os}")
    if arch_aliases.get(actual_arch, actual_arch) != arch_aliases.get(expected_arch, expected_arch):
        reasons.append(f"architecture is {actual_arch}, expected {profile.platform.architecture}")
    if cuda.device_count != profile.accelerator.count:
        reasons.append(
            f"CUDA device count is {cuda.device_count}, expected {profile.accelerator.count}"
        )
    if cuda.gpus:
        gpu = cuda.gpus[0]
        if not re.search(profile.accelerator.name_regex, gpu.name):
            reasons.append(f"GPU {gpu.name!r} does not match the profile name")
        if gpu.compute_capability != profile.accelerator.expected_compute_capability:
            expected_capability = profile.accelerator.expected_compute_capability
            reasons.append(
                f"compute capability is {gpu.compute_capability}, expected {expected_capability}"
            )
        minimum_bytes = int(profile.accelerator.nominal_total_vram_gib * 1024**3 * 0.98)
        if gpu.total_vram_bytes < minimum_bytes:
            reasons.append("reported total VRAM is below the nominal profile capacity")
        if profile.precision.require_bf16_runtime_check and not gpu.bf16_supported:
            reasons.append("BF16 is not supported by the active PyTorch runtime")
    return ProfileMatch(
        requested_id=profile.id,
        matched=not reasons,
        support_status=profile.support_status,
        reasons=reasons,
    )


def collect_doctor_report(
    profile: HardwareProfile | None = None,
    *,
    strict_profile: bool = False,
    torch_module: Any | None = None,
) -> DoctorReport:
    """Inspect the environment without downloading models or changing runtime state."""

    dependencies = _dependencies()
    if torch_module is None and dependencies["torch"].installed:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, OSError):
            torch_module = None
    cuda = _cuda_status(torch_module)
    match = _profile_match(profile, cuda)
    missing = [name for name, item in dependencies.items() if item.required and not item.installed]
    messages: list[str] = []
    if missing:
        messages.append(f"missing required dependencies: {', '.join(sorted(missing))}")
    if not cuda.available:
        messages.append("no usable CUDA device is visible")
    if profile and not match.matched:
        messages.extend(match.reasons)
    if profile and match.matched and profile.support_status != "validated":
        messages.append(f"hardware profile {profile.id} is {profile.support_status}")

    if strict_profile and profile and not match.matched:
        status: Literal["usable", "warning", "unusable", "profile_mismatch"] = "profile_mismatch"
        exit_code: Literal[0, 2, 3, 4] = 4
    elif missing or not cuda.available:
        status = "unusable"
        exit_code = 3
    elif profile is None or not match.matched or profile.support_status != "validated":
        status = "warning"
        exit_code = 2
    else:
        status = "usable"
        exit_code = 0

    cache = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    pytorch_version = dependencies["torch"].version
    tf32_available = bool(cuda.gpus and int(cuda.gpus[0].compute_capability.split(".")[0]) >= 8)
    return DoctorReport(
        status=status,
        exit_code=exit_code,
        os=platform.system().lower(),
        architecture=platform.machine().lower(),
        python_version=platform.python_version(),
        pytorch_version=pytorch_version,
        dependencies=dependencies,
        cuda=cuda,
        tf32_available=tf32_available,
        huggingface_cache=cache,
        docker=_docker_status(),
        profile=match,
        messages=messages,
    )
