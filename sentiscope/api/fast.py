#API imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml, load_pipeline, \
    load_tokenizer, preprocess_dl
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict, load_ml_model, load_dl_model

#Post request import
from pydantic import BaseModel

#Initialize FastAPI
app = FastAPI()

#Class for post request text
class Text(BaseModel):
    text: str

#Store models and preprocessing tools once loaded -> Speed up future requests
app.state.model_ml = load_ml_model('local')
app.state.model_dl = load_dl_model()
app.state.tokenizer = load_tokenizer()
app.state.pipeline = load_pipeline('local')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#Endpoint to use Deep Learning model
@app.get("/bert")
def bert_predict(review):
    """
    Make a prediction for the sentiment of a review using the bert-tiny model.

    The format of the url should be like this:

    http://127.0.0.1:8000/bert?review=This+is+a+bad+review
    """

    #Tokenize input
    X_pred = preprocess_dl(X=review, tokenizer=app.state.tokenizer)

    #Get a sentiment prediction
    prediction = app.state.model_dl.predict(X_pred)

    #Get the actual label out of the predictions
    logits = prediction.logits
    y_pred = np.argmax(logits, axis=1).tolist()

    #Convert class to natural language
    if y_pred[0] == 0:
        return {
            "Sentiment" : "Negative"
        }
    elif y_pred[0] == 1:
        return{
            "Sentiment" : "Positive"
        }
    else:
        return{
            "Sentiment" : "No output"
        }


#Define /predict endpoint
@app.get("/predict")
def predict_sentiment(review):
    """Make a prediction for the sentiment of a single review using a logisitc
    regression model.

    The format of the url should be like this:

    http://127.0.0.1:8000/predict?review=This+is+a+bad+review
    """

    #Preprocess the review
    X_pred = preprocess_ml(review,app.state.pipeline)

    #Predict the sentiment of the review
    prediction = predict(X_pred, app.state.model_ml)

    #Extract vectorizer for visualization data
    vectorizer = app.state.pipeline['vectorizer']

    #Extract data from model and vectorizer
    coefs = app.state.model_ml.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    input_indices = X_pred.nonzero()[1]
    tfidf_values = X_pred.toarray()[0][input_indices]
    input_tokens = [feature_names[i] for i in input_indices]
    word_coefs = coefs[input_indices]

    # Compute word contributions
    contributions = tfidf_values * word_coefs
    contrib_dict = dict(zip(input_tokens, contributions))

    # Sort contributions to find top positives and negatives
    sorted_items = sorted(contrib_dict.items(), key=lambda x: x[1])
    top_negative = [w for w, _ in sorted_items[:2]]
    top_positive = [w for w, _ in sorted_items[-2:]]

    #Turn predicted label to readable text and also return vis. data
    if prediction == -1:
        return {
            "Sentiment": "Negative",
            "contributions": contrib_dict,
            "top_positive": top_positive,
            "top_negative": top_negative
            }
    elif prediction == 0:
        return {
            "Sentiment": "Positive",
            "contributions": contrib_dict,
            "top_positive": top_positive,
            "top_negative": top_negative
            }
    else:
        return {"Sentiment": "No output"}


#TODO:finish this endpoint
@app.post("/text")
def receive_text(my_text: Text):
    body = my_text.text

    #Preprocess the review
    X_pred = preprocess_ml(body,app.state.pipeline)

    #Predict the sentiment of the review
    prediction = predict(X_pred, app.state.model)

    vectorizer = app.state.pipeline['vectorizer']

    coefs = app.state.model_ml.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    input_indices = X_pred.nonzero()[1]
    tfidf_values = X_pred.toarray()[0][input_indices]
    input_tokens = [feature_names[i] for i in input_indices]
    word_coefs = coefs[input_indices]

    # Compute word contributions
    contributions = tfidf_values * word_coefs
    contrib_dict = dict(zip(input_tokens, contributions))

    # Sort contributions to find top positives and negatives
    sorted_items = sorted(contrib_dict.items(), key=lambda x: x[1])
    top_negative = [w for w, _ in sorted_items[:2]]
    top_positive = [w for w, _ in sorted_items[-2:]]

    #Turn predicted label to readable text
    if prediction == -1:
        return {
            "Sentiment": "Negative",
            "contributions": contrib_dict,
            "top_positive": top_positive,
            "top_negative": top_negative
            }
    elif prediction == 0:
        return {
            "Sentiment": "Positive",
            "contributions": contrib_dict,
            "top_positive": top_positive,
            "top_negative": top_negative
            }
    else:
        return {"Sentiment": "No output"}

#Root endpoint
@app.get("/")
def root():
    return {'Greeting':'''Welcome to the SentiScope api.Use the
            /predict endpoint for predictions!'''}
