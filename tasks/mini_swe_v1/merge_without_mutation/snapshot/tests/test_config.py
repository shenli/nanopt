import unittest

from src.config import merge_config


class ConfigTests(unittest.TestCase):
    def test_override_wins(self) -> None:
        self.assertEqual(merge_config({"timeout": 5}, {"timeout": 8}), {"timeout": 8})

    def test_defaults_are_retained(self) -> None:
        self.assertEqual(merge_config({"a": 1}, {"b": 2}), {"a": 1, "b": 2})
