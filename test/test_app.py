"""
Tests for the prediction API in the Flask application.

This module contains a set of unit tests for the `/predict` endpoint in the Flask application. The tests cover various scenarios, including positive and negative sentiment predictions, as well as error cases such as empty or missing input text.

The `client` fixture is used to create a test client for the Flask application, which is used to make HTTP requests to the `/predict` endpoint.
"""
import pytest
from flask import Flask, json
from app import app, predict

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_positive(client):
    data = {'text': 'This movie was amazing! I loved it.'}
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200
    assert json.loads(response.data)['sentiment'] == 'positive'

def test_predict_negative(client):
    data = {'text': 'This movie was terrible. I hated it.'}
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200
    assert json.loads(response.data)['sentiment'] == 'negative'

def test_predict_empty_text(client):
    data = {'text': ''}
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 400

def test_predict_missing_text(client):
    data = {}
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 400

def test_predict_invalid_request(client):
    response = client.post('/predict', data='invalid', content_type='application/json')
    assert response.status_code == 400
