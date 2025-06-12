#Data
import pandas as pd

#Basic cleaning
import string
import re

#Saving the pipeline
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

    print("CLEANING!")

    #Turn input into pandas Series
    if type(X) != pd.core.series.Series:
        X = pd.Series(X)

    path_to_pipeline = "preprocessing_pipelines/preproc_pipeline_ml.pkl"



    # Load the pipeline using Pickle
    with open(path_to_pipeline, 'rb') as file:
        preproc_pipeline = pickle.load(file)

    X_processed = preproc_pipeline.transform(X)

    return X_processed
