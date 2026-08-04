from __future__ import annotations

import importlib.metadata

import nanopt


def test_import_and_distribution_versions_match() -> None:
    assert nanopt.__version__ == "0.3.0"
    assert importlib.metadata.version("nanopt") == nanopt.__version__
