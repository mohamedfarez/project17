"""
This script trains a Multinomial Naive Bayes classifier on a dataset of text reviews and their associated sentiment labels (positive or negative). The script performs the following steps:

1. Imports the necessary libraries and modules.
2. Loads the dataset from a CSV file.
3. Cleans the text data using the `clean_text` function.
4. Converts the text data to a numerical representation using TF-IDF vectorization.
5. Splits the data into training and testing sets.
6. Trains the Multinomial Naive Bayes classifier on the training data.
7. Evaluates the model's performance on the testing data, printing the accuracy and classification report.
8. Saves the trained model and vectorizer to disk using the `joblib` library.
"""
#import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scripts.clean_text import clean_text
import joblib


#import data
data_path = 'data/dataset.csv'
df = pd.read_csv(data_path)

#cleaning data
df['cleaned_review'] = df['review'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review']).toarray()
#make pos =1 and nev=0
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# bulid model
model = MultinomialNB()

# train model
model.fit(X_train, y_train)
#preduict
y_pred = model.predict(X_test)
 

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

#save model trained
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
