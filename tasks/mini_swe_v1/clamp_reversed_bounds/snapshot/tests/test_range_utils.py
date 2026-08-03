import unittest

from src.range_utils import clamp


class ClampTests(unittest.TestCase):
    def test_inside_range_is_unchanged(self) -> None:
        self.assertEqual(clamp(4, 1, 9), 4)

    def test_outside_range_is_clamped(self) -> None:
        self.assertEqual(clamp(12, 1, 9), 9)


if __name__ == "__main__":
    unittest.main()
