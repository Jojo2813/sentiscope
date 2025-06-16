import joblib
from google.cloud import storage

#Import for loading the bert model
from transformers import TFAutoModelForSequenceClassification

#Parameters
from sentiscope.params import *

def load_dl_model():
    """
    Method to load pre trained bert-tiny model
    """

    #Load model from local file and return it
    model = TFAutoModelForSequenceClassification.\
        from_pretrained("./models/bert_tiny_180k")

    return model

def load_ml_model(target):
    """
    Method to load pre trained logreg model
    """
    if target == 'local':
        #Load the model from local
        model = joblib.load(\
            "./models/logreg_full.pkl")
    elif target == 'gcs':
        #Load model from gcs
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB)
        blob.download_to_filename("./models/logreg_full.pkl")

        model = joblib.load("./models/logreg_full.pkl")
    return model

def predict(input, model):
    """
    Function that loads trained logistic regression model
    and predicts label of given review.
    Input: Vectorized user input
    Output: label (-1: bad, 0 : good)
    """

    #Make a prediction
    prediction = model.predict(input)

    return prediction
