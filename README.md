# project17
movie review 
by: mohamed fares

you should creat new enev 
use this code in your terminal : python -m venv venv
after that actevate the enve : .\venv\Scripts\activate




in app.py
Provides a Flask API endpoint for predicting the sentiment of a given text input.

The `/predict` endpoint accepts a JSON payload with a `text` field containing the text to be analyzed. The text is first cleaned using the `clean_text` function from the `scripts.clean_text` module, then vectorized using the `vectorizer` object loaded from a pickle file. The vectorized data is then passed to the `model` object (also loaded from a pickle file) to make the sentiment prediction. The predicted sentiment is returned as a JSON response with a `sentiment` field set to either `'positive'` or `'negative'`.

in clean_text.py
Cleans the input text by performing the following operations:
- Removes HTML tags
- Removes non-alphanumeric characters
- Converts the text to lowercase
- Removes numeric characters
- Removes English stopwords

Args:
    text (str): The input text to be cleaned.

Returns:
    str: The cleaned text.


in model.py

This script trains a Multinomial Naive Bayes classifier on a dataset of text reviews and their associated sentiment labels (positive or negative). The script performs the following steps:

1. Imports the necessary libraries and modules.
2. Loads the dataset from a CSV file.
3. Cleans the text data using the `clean_text` function.
4. Converts the text data to a numerical representation using TF-IDF vectorization.
5. Splits the data into training and testing sets.
6. Trains the Multinomial Naive Bayes classifier on the training data.
7. Evaluates the model's performance on the testing data, printing the accuracy and classification report.
8. Saves the trained model and vectorizer to disk using the `joblib` library.

