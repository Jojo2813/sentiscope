import pandas as pd
import numpy as np
import os

#Basic cleaning
import string
import re

#ML tokenizing, lemmatizing and vectorizing
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def basic_cleaning(text):
    """
    Function which takes a string and cleans it to get the string ready for
    future preprocessing. This is a universal step which the data will always
    undergo.
    Input: String
    Output: String
    """

    #No whitespaces in beginning or end
    text = text.strip()

    #lowercase
    text= text.lower()

    #remove numbers
    text = re.sub(r'\b\d+\b', '', text)

    #Removing punctuation
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)

    # Tokenizing
    tokenized = word_tokenize(text)

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]

    text = " ".join(lemmatized)
    return text

def preprocess_series(X):
        """
        Helper function for pipeline
        This function always has to be available in the namespace when loading
        the pipeline.
        """
        return[basic_cleaning(text) for text in X]
