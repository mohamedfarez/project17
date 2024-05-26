"""
Loads and preprocesses the IMDB movie review dataset.

Args:
    data_dir (str): The directory containing the IMDB movie review data.

Returns:
    pandas.DataFrame: A DataFrame containing the movie reviews and their sentiment labels.
"""
#import library

import os
import pandas as pd

def load_imdb_data(data_dir):
    data = {'review': [], 'sentiment': []}
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(data_dir, sentiment)
        for filename in os.listdir(sentiment_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(sentiment_dir, filename), 'r', encoding='utf-8') as f:
                    data['review'].append(f.read())
                    data['sentiment'].append('positive' if sentiment == 'pos' else 'negative')
    return pd.DataFrame(data)

data_dir = r'.....'  #delete ... and add your path  etc (data/data_dir)

df = load_imdb_data(data_dir)


df.to_csv('data/dataset.csv', index=False)
