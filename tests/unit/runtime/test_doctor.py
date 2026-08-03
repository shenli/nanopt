from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from nanopt.config.loader import ConfigRepository
from nanopt.runtime.doctor import collect_doctor_report


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="NVIDIA GeForce RTX 4070 Ti SUPER", total_memory=16 * 1024**3)

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return 15 * 1024**3, 16 * 1024**3

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return 8, 9

    def is_bf16_supported(self, *, including_emulation: bool = False) -> bool:
        assert including_emulation is False
        return True


class FakeTorch:
    __version__ = "2.7.0"
    version = SimpleNamespace(cuda="12.8")
    cuda = FakeCuda()


class DriverReservedCuda(FakeCuda):
    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        # Measured through PyTorch on the reference card. The product is sold as
        # 16 GB, but the CUDA runtime does not expose every byte to applications.
        return SimpleNamespace(name="NVIDIA GeForce RTX 4070 Ti SUPER", total_memory=16_714_694_656)


class DriverReservedTorch(FakeTorch):
    cuda = DriverReservedCuda()


def test_cpu_only_report_is_machine_readable(project_root: Path) -> None:
    report = collect_doctor_report(
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=None)
        )
    )
    assert report.exit_code == 3
    assert report.status == "unusable"
    assert report.cuda.gpus == []
    value = report.model_dump(mode="json")
    schema = json.loads((project_root / "specs/schemas/doctor_report.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(value)


def test_matching_unvalidated_gpu_returns_warning(project_root: Path) -> None:
    profile = ConfigRepository(project_root / "configs").hardware("rtx_4070_ti_super_16gb")
    report = collect_doctor_report(profile, torch_module=FakeTorch())
    # The mocked GPU matches; the host OS may not, so use the profile reasons to distinguish it.
    hardware_reasons = [
        reason
        for reason in report.profile.reasons
        if not reason.startswith(("OS is", "architecture is"))
    ]
    assert hardware_reasons == []
    assert report.cuda.gpus[0].bf16_supported
    assert report.tf32_available


def test_profile_match_allows_measured_driver_reserved_vram(project_root: Path) -> None:
    profile = ConfigRepository(project_root / "configs").hardware("rtx_4070_ti_super_16gb")
    report = collect_doctor_report(profile, torch_module=DriverReservedTorch())
    assert "reported total VRAM is below the nominal profile capacity" not in report.profile.reasons


def test_strict_profile_mismatch_has_dedicated_exit(project_root: Path) -> None:
    profile = ConfigRepository(project_root / "configs").hardware("rtx_4070_ti_super_16gb")
    report = collect_doctor_report(
        profile,
        strict_profile=True,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=None)
        ),
    )
    assert report.exit_code == 4
    assert report.status == "profile_mismatch"
