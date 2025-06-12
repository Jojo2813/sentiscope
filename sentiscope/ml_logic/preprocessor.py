#Data
import pandas as pd

#Basic cleaning
import string
import re

#Loading the saved pipeline
import pickle

#Helper function
from sentiscope.ml_logic.utils import preprocess_series


def preprocess_ml(X):
    """
    Function that uses a pipeline to preprocess natural language to get it
    ready for a classical machine learning model. Data gets cleaned and
    vectorized using a TF-IDF vectorizer from sklearn.

    Input: Pandas Series
    Output: Sparse Matrix (float64)
    """

    #Turn input into pandas Series, that's what the pipeline expects
    if type(X) != pd.core.series.Series:
        X = pd.Series(X)

    #TODO: store path in .env file?
    path_to_pipeline = "preprocessing_pipelines/preproc_pipeline_ml.pkl"

    # Load the pipeline using Pickle
    with open(path_to_pipeline, 'rb') as file:
        preproc_pipeline = pickle.load(file)

    #Let the pipeline trnsform the input to get it ready for the model
    X_processed = preproc_pipeline.transform(X)

    return X_processed
