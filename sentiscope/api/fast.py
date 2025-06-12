#API imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict, load_model

#Initialize FastAPI
app = FastAPI()

#Store model once loaded -> Speed up future requests
app.state.model = load_model()

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

    #Preprocess the review
    X_pred = preprocess_ml(review)

    #Predict the sentiment of the review
    prediction = predict(X_pred)

    #Turn predicted label to readable text
    if prediction == -1:
        return {"Sentiment": "Negative"}
    elif prediction == 0:
        return {"Sentiment": "Positive"}
    else:
        return {"Sentiment": "No output"}

#Root endpoint
@app.get("/")
def root():
    return {'Greeting':'''Welcome to the SentiScope api.Use the
            /predict endpoint for predictions!'''}
