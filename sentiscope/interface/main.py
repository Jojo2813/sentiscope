#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml, load_pipeline
#This function must be included for the pipeline to work
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict, load_model
from sentiscope.params import *

#Testing model and preprocessing
def test_package(text):

    pipe = load_pipeline(target='local')

    model = load_model(target='gcs')

    #Preprocess input
    vector_text = preprocess_ml(text, pipe=pipe)

    #Predict sentiment
    prediction = predict(vector_text,model=model)

    #Check sentiment
    print(prediction)

if __name__ == "__main__":
    #Insert any string you want
    text = "I love that I hate it"
    test_package(text)
