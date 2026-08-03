import unittest

from src.config import merge_config


class HiddenConfigTests(unittest.TestCase):
    def test_defaults_are_not_mutated(self) -> None:
        defaults = {"timeout": 5}
        merge_config(defaults, {"timeout": 8})
        self.assertEqual(defaults, {"timeout": 5})

    def test_overrides_are_not_mutated(self) -> None:
        overrides = {"timeout": 8}
        merge_config({"timeout": 5}, overrides)
        self.assertEqual(overrides, {"timeout": 8})

    def test_result_is_fresh_for_empty_override(self) -> None:
        defaults = {"timeout": 5}
        self.assertIsNot(merge_config(defaults, {}), defaults)
