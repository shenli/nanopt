import unittest

from src.chunks import chunks


class ChunkTests(unittest.TestCase):
    def test_even_chunks(self) -> None:
        self.assertEqual(chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_remainder(self) -> None:
        self.assertEqual(chunks([1, 2, 3], 2), [[1, 2], [3]])
