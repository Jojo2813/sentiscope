import pandas as pd
import numpy as np
import os

#Basic cleaning
import string
import re

#ML tokenizing, lemmatizing and vectorizing
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

#Saving the pipeline
import pickle

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

    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)

    # Tokenizing
    tokenized = word_tokenize(text)
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
    text = " ".join(lemmatized)
    return text


def preprocess_ml(X):
    """
    Function that uses a pipeline to preprocess natural language to get it
    ready for a classical machine learning model. Data gets cleaned and
    vectorized using a TF-IDF vectorizer from sklearn.

    Input: Pandas Series
    Output: Sparse Matrix (float64)
    """
    path_to_pipeline = "preprocessing_pipelines/preproc_pipeline_ml.pkl"



    # Load the pipeline using Pickle
    with open(path_to_pipeline, 'rb') as file:
        preproc_pipeline = pickle.load(file)

    X_processed = preproc_pipeline.transform(X)

    return X_processed
