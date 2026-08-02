"""Typed configuration loading and deterministic resolution."""

from nanopt.config.models import ResolvedConfig
from nanopt.config.resolver import resolve_config

__all__ = ["ResolvedConfig", "resolve_config"]
