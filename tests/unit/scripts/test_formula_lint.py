from __future__ import annotations

from pathlib import Path

from scripts.lint_formulas import lint_file


def test_formula_lint_accepts_dollar_math_and_literal_code(tmp_path: Path) -> None:
    path = tmp_path / "good.md"
    path.write_text("Inline $x$ and `\\(literal\\)`\n\n```text\n\\[literal\\]\n```\n")
    assert lint_file(path) == []


def test_formula_lint_rejects_raw_delimiters_and_unclosed_fences(tmp_path: Path) -> None:
    raw = tmp_path / "raw.md"
    raw.write_text("Bad \\(x\\)\n")
    fence = tmp_path / "fence.md"
    fence.write_text("```python\n")
    assert "unsupported formula delimiter" in lint_file(raw)[0]
    assert "unclosed Markdown code fence" in lint_file(fence)[0]
