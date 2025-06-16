#Data
import pandas as pd

#Basic cleaning
import string
import re

#Loading the saved pipeline
import pickle
import dill

#Helper function
from sentiscope.ml_logic.utils import preprocess_series

#GCS imports
from google.cloud import storage
from sentiscope.params import *

from transformers import AutoTokenizer

def load_tokenizer():
    tokenizer = AutoTokenizer.\
        from_pretrained("./models/tokenizer_bert_tiny_180k")

    return tokenizer

def preprocess_dl(X, tokenizer):

    if type(X) != pd.core.series.Series:
        X = pd.Series(X)

    tokenized_tensor = tokenizer(X.tolist(), \
        max_length=400, padding = "max_length", \
            truncation = True, return_tensors="tf")

    return tokenized_tensor

def load_pipeline(target):

    if target == 'local':
        #Load the model from local
        with open ("./models/preproc_pipeline_ml_2.pkl", \
            'rb') as file:
            pipe = dill.load(file)
    elif target == 'gcs':
        #Load model from gcs
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(PIPE_BLOB)
        blob.download_to_filename("./preprocessing_pipelines/preproc_pipeline_ml.pkl")

        with open ("./preprocessing_pipelines/preproc_pipeline_ml.pkl", \
            'rb') as file:
            pipe = pickle.load(file)

        # Ensure full recursive serialization
        dill.settings['recurse'] = True
        with open("./models/preproc_pipeline_ml_2.pkl", "wb") as f:
            dill.dump(pipe, f)

    return pipe


def preprocess_ml(X,pipe):
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

    #Let the pipeline trnsform the input to get it ready for the model
    X_processed = pipe.transform(X)

    return X_processed
