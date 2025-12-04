import re
import string 
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK downloads
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK is working")

if __name__ == "__main__":
    download_nltk_resources()
    print("✅ NLTK resources downloaded.")

def preprocess_text(text, remove_stopwords=False, remove_numbers=False):
    # lowercase the text
    text = text.lower()

    # Remove non-ASCII characters (special symbols, emojis, etc.)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # remove the numbers 
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # remove the punctuatuion
    text = text.translate(str.maketrans("","", string.punctuation))

    # Tokenize words
    tokens = word_tokenize(text) 

    # remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # apply lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(lemmatized_tokens)