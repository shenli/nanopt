import unittest

from src.boolean import parse_bool


class BooleanTests(unittest.TestCase):
    def test_true_literal(self) -> None:
        self.assertTrue(parse_bool("true"))

    def test_false_literal(self) -> None:
        self.assertFalse(parse_bool("false"))
