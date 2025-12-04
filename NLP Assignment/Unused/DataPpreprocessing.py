import re
import pdfreader
import string 
import streamlit as st
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Already move to extractfile_utils
# def extract_text_from_pdf(pdf_path):
#     reader = pdfreader(pdf_path)
#     text = ""

#     # Loop through all pages and extract text
#     for page in reader.pages:
#         try:
#             text += page.extract_text() + "\n"
#         except:
#             continue

#     return text

# Initialize the stopwords and stemmer
stopwords = set(stopwords.words('english'))

def preprocess_text(text):

    # lowercase the text
    text = text.lower()

    # remove the special characters (remove the non-words charcter)
    text = re.sub(r'\W', '', text)

    # remove the numbers 
    text = re.sub(r'\d+', '', text)

    # remove the punctuatuion
    text = text.translate(str.maketrans("","", string.punctuation))

    # Tokenize words
    tokens = word_tokenize.tokenize(text)

    # remove stop words
    stopwords = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords]

    # apply stemming
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return lemmatized_tokens

# This one maybe need to put inside the main 
# st.title("📄 PDF Text Extractor & Tokenizer")

# uploaded_pdf = st.file_uploader("Drop your PDF here", type=["pdf"])

# if uploaded_pdf:
#     with st.spinner("Processing pdf..."):
#         raw_text = extract_text_from_pdf(uploaded_pdf)
#         tokens = preprocess_text(raw_text)
#         cleaned_text = " ".join(tokens)

#         st.subheader("🔍 Extracted & Tokenized Text")
#         st.text_area("Cleaned Text", cleaned_text, height = 300)

#         st.subheader("📦 Tokens")
#         st.write(tokens)


