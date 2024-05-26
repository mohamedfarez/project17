"""
Tests for the `clean_text` function in the `scripts.clean_text` module.

The `TestCleanText` class contains unit tests that verify the behavior of the
`clean_text` function, which is responsible for cleaning and preprocessing text
data. The tests cover the following functionality:

- Removing HTML tags from the input text
- Removing non-alphanumeric characters from the input text
- Converting the input text to lowercase
- Removing numeric characters from the input text
- Removing stopwords from the input text
- Handling an empty input string

These tests ensure that the `clean_text` function behaves as expected and
produces the desired output for a variety of input scenarios.
"""
import unittest
from scripts.clean_text import clean_text

class TestCleanText(unittest.TestCase):

    def test_remove_html_tags(self):
        text = "This is a <b>test</b> string with <i>HTML</i> tags."
        expected_output = "This is a test string with HTML tags."
        self.assertEqual(clean_text(text), expected_output)

    def test_remove_non_alphanumeric(self):
        text = "This is a test string with !@#$%^&*() characters."
        expected_output = "This is a test string with  characters"
        self.assertEqual(clean_text(text), expected_output)

    def test_convert_to_lowercase(self):
        text = "This Is A Test String With Mixed Case."
        expected_output = "this is a test string with mixed case"
        self.assertEqual(clean_text(text), expected_output)

    def test_remove_numeric_characters(self):
        text = "This is a test string with 123 numbers."
        expected_output = "this is a test string with  numbers"
        self.assertEqual(clean_text(text), expected_output)

    def test_remove_stopwords(self):
        text = "This is a test string with some stopwords."
        expected_output = "test string stopwords"
        self.assertEqual(clean_text(text), expected_output)

    def test_empty_string(self):
        text = ""
        expected_output = ""
        self.assertEqual(clean_text(text), expected_output)

if __name__ == '__main__':
    unittest.main()
