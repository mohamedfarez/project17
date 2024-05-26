"""
This module contains unit tests for the machine learning model used in the application.

The `TestModel` class inherits from `unittest.TestCase` and defines several test methods to ensure the correct functionality of the model:

- `test_clean_text`: Verifies that the `clean_text` function correctly removes punctuation from the input text.
- `test_vectorizer`: Checks that the TF-IDF vectorizer is properly configured to use a maximum of 5,000 features.
- `test_model_fit`: Ensures that the Multinomial Naive Bayes model can be successfully trained on the provided data.
- `test_model_predict`: Verifies that the trained model can make predictions on the test data.

The `setUp` method is used to prepare the test data, including creating a sample DataFrame, transforming the text data using the TF-IDF vectorizer, and initializing the Multinomial Naive Bayes model.
"""
import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scripts.model import clean_text

class TestModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'review': ['This is a great product', 'I hate this product', 'The product is okay'],
            'sentiment': ['positive', 'negative', 'positive']
        })
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X = self.vectorizer.fit_transform(self.data['review'].apply(clean_text)).toarray()
        self.y = self.data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
        self.model = MultinomialNB()

    def test_clean_text(self):
        cleaned_text = clean_text('This is a test! with punctuation?')
        self.assertEqual(cleaned_text, 'this is a test with punctuation')

    def test_vectorizer(self):
        self.assertEqual(self.X.shape[1], 5000)

    def test_model_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsInstance(self.model, MultinomialNB)

    def test_model_predict(self):
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

if __name__ == '__main__':
    unittest.main()
