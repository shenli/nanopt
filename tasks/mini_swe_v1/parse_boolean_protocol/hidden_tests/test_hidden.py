import unittest

from src.boolean import parse_bool


class HiddenBooleanTests(unittest.TestCase):
    def test_case_and_whitespace(self) -> None:
        self.assertTrue(parse_bool("  YeS "))

    def test_zero(self) -> None:
        self.assertFalse(parse_bool("0"))

    def test_unknown_string(self) -> None:
        with self.assertRaises(ValueError):
            parse_bool("maybe")

    def test_non_string(self) -> None:
        with self.assertRaises(TypeError):
            parse_bool(1)  # type: ignore[arg-type]
