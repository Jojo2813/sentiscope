#API imports
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

#Preprocessing
from sentiscope.ml_logic.preprocessor import load_pipeline, \
    load_tokenizer
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import load_ml_model, load_dl_model

#Post request import
from pydantic import BaseModel

#imports for preprocessing, predictions and visualization parameters
from sentiscope.ml_logic.explain import \
    create_predict_fn, explain_with_lime, explain_ml

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

    #Define function needed to get bert prediction and analysis
    predict_proba_fn = \
        create_predict_fn(app.state.model_dl, app.state.tokenizer)

    #Get sentiment and visualization params
    explanation = explain_with_lime(review, predict_proba_fn)

    return explanation


#Define /predict endpoint
@app.get("/predict")
def predict_sentiment(review):
    """Make a prediction for the sentiment of a single review using a logisitc
    regression model.

    The format of the url should be like this:

    http://127.0.0.1:8000/predict?review=This+is+a+bad+review
    """

    #Call function to preprocess and predict review
    return explain_ml(review=review, \
        model = app.state.model_ml, pipeline=app.state.pipeline,target='single')


@app.post("/text_ml")
def receive_ml_text(my_text: Text):
    """
    Endpoint to make a prediction via POST.
    """
    review = my_text.text

    return explain_ml(review=review, \
        model = app.state.model_ml, pipeline=app.state.pipeline,target='single')


@app.post("/text_dl")
def receive_dl_text(my_text: Text):
    """
    Endpoint to accept Post request and make prediction with BERT
    """

    review = my_text.text

    #Define function needed to get bert prediction and analysis
    predict_proba_fn = \
        create_predict_fn(app.state.model_dl, app.state.tokenizer)

    #Get sentiment and visualization params
    explanation = explain_with_lime(review, predict_proba_fn)

    return explanation

#Endpoint for batch prediction, expects csv
@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...)):

    #Check if file is a csv
    if not file.filename.endswith('.csv'):
        return JSONResponse(content={"error":"Only CSV files are supported"},\
            status_code=400)

    #Turn csv to dataframe
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    #Make sure there is a text column
    if "text" not in df.columns:
        return JSONResponse(\
            content={"error": "CSV must contain a 'text' column"},\
                status_code=400)

    # Apply prediction
    predictions = explain_ml(\
        df['text'],app.state.model_ml, app.state.pipeline,target='batch')
    df["predicted_sentiment"] = predictions
    df["predicted_sentiment"] = \
        df["predicted_sentiment"].map({-1: "Negative", 0: "Positive"})

    # Return as JSON
    return df.to_dict(orient="records")

#Root endpoint
@app.get("/")
def root():
    return {'Greeting':'''Welcome to the SentiScope api. Use the
            /docs endpoint to see the different endpoints!'''}
