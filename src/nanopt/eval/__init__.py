"""Strict response parsing and deterministic exact-answer verification."""

from nanopt.eval.parser import ParseResult, parse_answer
from nanopt.eval.verifier import VerificationResult, verify_task_response

__all__ = ["ParseResult", "VerificationResult", "parse_answer", "verify_task_response"]
