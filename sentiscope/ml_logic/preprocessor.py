import pandas as pd
import numpy as np
import os

#Basic cleaning
import string
import re

#ML tokenizing, lemmatizing and vectorizing
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Pipeline imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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

    If there is no local pipeline stored, it will create a new one and save it.

    Input: Pandas Series
    Output: Sparse Matrix (float64)
    """
    path_to_pipeline = "preprocessing_pipelines/preproc_pipeline_ml.pkl"

    #Check if pipeline already exists
    if os.path.exists(path_to_pipeline):

        # Load the pipeline using Pickle
        with open(path_to_pipeline, 'rb') as file:
            preproc_pipeline = pickle.load(file)

        X_processed = preproc_pipeline.transform(X)

        return X_processed
     #Else create a new one and save it locally
    else:
        def preprocess_series(X):
            """
            Helper function to include a custom function into the pipeline.
            """
            return[basic_cleaning(text) for text in X]

        #Including the cleaning function into a Functiontransformer
        transformer = FunctionTransformer(func=preprocess_series)

        #Define a Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features = 10000)

        #Building the pipeline with cleaning and then vectorizing
        preproc_pipe = Pipeline([('cleaning', transformer), \
            ('vectorizer', vectorizer)])

        #Fit pipeline and transform data
        X_processed = preproc_pipe.fit_transform(X)

        # Export Pipeline as pickle file
        with open(path_to_pipeline, "wb") as file:
            pickle.dump(preproc_pipe, file)

        return X_processed
