"""Strict extraction and canonicalization of one final answer field."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from nanopt.data.schemas import AnswerType

ParseStatus = Literal[
    "valid",
    "response_too_long",
    "missing_answer",
    "multiple_answers",
    "malformed_answer",
    "trailing_content",
    "invalid_value",
]

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_ANSWER_TAG_LIKE = re.compile(r"<\s*/?\s*answer\b[^>]*>", flags=re.IGNORECASE)
_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)")
_RATIONAL_PATTERN = re.compile(r"(-?(?:0|[1-9][0-9]*))/([1-9][0-9]*)")
ANSWER_CLOSE_TAG = "</answer>"


def answer_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Tokenize the task protocol's closing answer tag without adding special tokens."""

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide encode for the answer stop sequence")
    values = encode(ANSWER_CLOSE_TAG, add_special_tokens=False)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, int) and value >= 0 for value in values)
    ):
        raise ValueError("tokenizer returned an invalid answer stop sequence")
    return tuple(values)


@dataclass(frozen=True)
class ParseResult:
    """Parser outcome kept separate from correctness verification."""

    status: ParseStatus
    canonical_answer: str | None = None
    raw_answer: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid"


def _invalid(status: ParseStatus, error: str) -> ParseResult:
    return ParseResult(status=status, error=error)


def _canonicalize(raw: str, answer_type: AnswerType) -> str | None:
    if answer_type == "integer":
        return raw if _INTEGER_PATTERN.fullmatch(raw) else None
    if answer_type == "rational":
        match = _RATIONAL_PATTERN.fullmatch(raw)
        if match is None:
            return None
        value = Fraction(int(match.group(1)), int(match.group(2)))
        return (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        )
    if not raw or any(ord(character) < 32 and character not in "\t" for character in raw):
        return None
    return raw


def parse_answer(
    response: str,
    *,
    answer_type: AnswerType,
    max_response_characters: int = 16_384,
    max_answer_characters: int = 128,
) -> ParseResult:
    """Parse exactly one lowercase final ``<answer>...</answer>`` field.

    Text before the answer field may contain a derivation. Only whitespace may follow the closing
    tag. Tag variants with different case, whitespace, or attributes are rejected rather than
    interpreted as additional answer channels. Integer and rational answers use narrow canonical
    grammars; floating-point spellings such as NaN and infinity are never accepted.
    """

    if max_response_characters <= 0 or max_answer_characters <= 0:
        raise ValueError("parser length limits must be positive")
    if len(response) > max_response_characters:
        return _invalid("response_too_long", "response exceeds parser character limit")

    exact_open_count = response.count("<answer>")
    exact_close_count = response.count("</answer>")
    tag_like = _ANSWER_TAG_LIKE.findall(response)
    if exact_open_count == 0 and exact_close_count == 0:
        return _invalid("missing_answer", "response does not contain an answer field")
    if exact_open_count > 1 or exact_close_count > 1:
        return _invalid("multiple_answers", "response contains multiple answer fields")
    if exact_open_count != 1 or exact_close_count != 1 or len(tag_like) != 2:
        return _invalid("malformed_answer", "answer tags are malformed or non-canonical")

    match = _ANSWER_PATTERN.search(response)
    if match is None:
        return _invalid("malformed_answer", "answer closing tag appears before opening tag")
    if response[match.end() :].strip():
        return _invalid("trailing_content", "non-whitespace content follows the final answer")
    raw_answer = match.group(1).strip()
    if len(raw_answer) > max_answer_characters or "<" in raw_answer or ">" in raw_answer:
        return _invalid("invalid_value", "answer value is too long or contains markup")
    canonical = _canonicalize(raw_answer, answer_type)
    if canonical is None:
        return _invalid("invalid_value", f"answer is not a valid {answer_type}")
    return ParseResult(
        status="valid",
        canonical_answer=canonical,
        raw_answer=raw_answer,
    )
