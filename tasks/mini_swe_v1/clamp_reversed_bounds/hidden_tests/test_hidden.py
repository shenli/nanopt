import unittest

from src.range_utils import clamp


class HiddenClampTests(unittest.TestCase):
    def test_reversed_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clamp(4, 9, 1)

    def test_equal_bounds_are_valid(self) -> None:
        self.assertEqual(clamp(99, 3, 3), 3)
