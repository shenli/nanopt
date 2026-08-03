import unittest

from src.chunks import chunks


class HiddenChunkTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(chunks([], 2), [])

    def test_zero_size(self) -> None:
        with self.assertRaises(ValueError):
            chunks([1], 0)

    def test_negative_size(self) -> None:
        with self.assertRaises(ValueError):
            chunks([1], -1)

    def test_non_integer_size(self) -> None:
        with self.assertRaises(TypeError):
            chunks([1], 1.5)  # type: ignore[arg-type]
