"""
Provides a Flask API endpoint for predicting the sentiment of a given text input.

The `/predict` endpoint accepts a JSON payload with a `text` field containing the text to be analyzed. The text is first cleaned using the `clean_text` function from the `scripts.clean_text` module, then vectorized using the `vectorizer` object loaded from a pickle file. The vectorized data is then passed to the `model` object (also loaded from a pickle file) to make the sentiment prediction. The predicted sentiment is returned as a JSON response with a `sentiment` field set to either `'positive'` or `'negative'`.
"""
#import library

from flask import Flask, request, jsonify
import joblib
from scripts.clean_text import clean_text

app = Flask(__name__)

#load model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    cleaned_data = clean_text(data)
    vectorized_data = vectorizer.transform([cleaned_data]).toarray()
    prediction = model.predict(vectorized_data)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
