import streamlit as st
import os
import numpy as np

from sklearn.feature_extraction.text import TfidVectorizer, CountVectorizer

# Import text preprocessing modules

from string import punctuation
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import regular expression
import re

import joblib
import warnings


warnings.filterwarnings("ignore")

# seeding

np.random.seed(123)

# load stop words

stop_words = stopwords.words("english")



# function to clean the text

@st.cache
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text with the option to remove stop_words and to lemmatize word

    # clean the text

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"https\S+", " link ", text)

    # Remove numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)

    # Remove punctuation from text

    text = "".join([c for c in text if c not in punctuation])

    # Optionally remove stop words

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally shorten words to their stem

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # return a list of words


    
# Create predictions

@st.cache
def make_predictions(review):

    # Clean the data 
    clean_review = text_cleaning(review)
    