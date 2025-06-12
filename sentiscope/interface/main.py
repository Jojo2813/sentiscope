#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml
#This function must be included for the pipeline to work
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict

#Testing model and preprocessing
def test_package(text):

    #Preprocess input
    vector_text = preprocess_ml(text)

    #Predict sentiment
    prediction = predict(vector_text)

    #Check sentiment
    print(prediction)

if __name__ == "__main__":
    #Insert any string you want
    text = "I love htat I hate it"
    test_package(text)
