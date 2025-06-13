#API imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml, load_pipeline
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict, load_model

from pydantic import BaseModel

#Initialize FastAPI
app = FastAPI()

class Text(BaseModel):
    text: str

#Store model once loaded -> Speed up future requests
app.state.model = load_model('local')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



#Define /predict endpoint
@app.get("/predict")
def predict_sentiment(review):
    """Make a prediction for the sentiment of a single review.

    The format of the url should be like this:

    http://127.0.0.1:8000/predict?review=This+is+a+bad+review
    """

    pipe = load_pipeline('local')

    #Preprocess the review
    X_pred = preprocess_ml(review,pipe)

    #Predict the sentiment of the review
    prediction = predict(X_pred, app.state.model)

    vectorizer = pipe['vectorizer']

    coefs = app.state.model.coef_[0]
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


@app.post("/text")
def receive_text(my_text: Text):
    body = my_text.text

    pipe = load_pipeline('local')

    #Preprocess the review
    X_pred = preprocess_ml(body,pipe)

    #Predict the sentiment of the review
    prediction = predict(X_pred, app.state.model)

    vectorizer = pipe['vectorizer']

    coefs = app.state.model.coef_[0]
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
