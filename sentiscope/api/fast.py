import sys
sys.path.append("/Users/johannesb/code/Jojo2813/SentiScope")

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentiscope.ml_logic.preprocessor import preprocess_ml
from sentiscope.ml_logic.utils import preprocess_series
from sentiscope.interface.main import test_package
from sentiscope.ml_logic.model import predict, load_model

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?review=I+love+it
@app.get("/predict")
def predict_sentiment(review):
    """Make a prediction for the sentiment of a single review"""

    X_pred = preprocess_ml(review)
    prediction = predict(X_pred)

    if prediction == -1:
        return {"Sentiment": "Negative"}
    elif prediction == 0:
        return {"Sentiment": "Positive"}
    else:
        return {"Sentiment": "No output"}

@app.get("/")
def root():
    return {'greeting':'Hello'}
