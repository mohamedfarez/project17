"""
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
"""
#import library

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower() 
    text = re.sub(r'\d+', '', text)  
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])   
    return text
