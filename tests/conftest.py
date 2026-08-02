from __future__ import annotations

from pathlib import Path

import pytest

from nanopt.config.loader import ConfigRepository


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config_repository(project_root: Path) -> ConfigRepository:
    return ConfigRepository(project_root / "configs")
