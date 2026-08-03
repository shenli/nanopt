import unittest

from src.slug import slugify


class HiddenSlugTests(unittest.TestCase):
    def test_collapses_mixed_separators(self) -> None:
        self.assertEqual(slugify("alpha -- beta___gamma"), "alpha-beta-gamma")

    def test_trims_edges(self) -> None:
        self.assertEqual(slugify("...Hello!"), "hello")

    def test_only_separators_is_empty(self) -> None:
        self.assertEqual(slugify("---"), "")
