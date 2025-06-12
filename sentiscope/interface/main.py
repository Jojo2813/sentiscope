#Preprocessing
from sentiscope.ml_logic.preprocessor import preprocess_ml
#This function must be included for the pipeline to work
from sentiscope.ml_logic.utils import preprocess_series

#Prediction
from sentiscope.ml_logic.model import predict

#Testing model and preprocessing
def test_package(text):

    vector_text = preprocess_ml(text)

    prediction = predict(vector_text)

    print(prediction)

if __name__ == "__main__":
    text = ["I would not buy again", "I hate it", "I love it"]
    test_package(text)
