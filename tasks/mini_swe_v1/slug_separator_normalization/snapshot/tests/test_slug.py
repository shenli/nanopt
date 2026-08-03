import unittest

from src.slug import slugify


class SlugTests(unittest.TestCase):
    def test_lowercases_words(self) -> None:
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_single_word(self) -> None:
        self.assertEqual(slugify("NanoPT"), "nanopt")
