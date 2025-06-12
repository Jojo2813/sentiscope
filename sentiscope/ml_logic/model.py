import joblib

def load_model():
    #Load the model
    model = joblib.load(\
        "/Users/johannesb/code/Jojo2813/SentiScope/model/logreg_full.pkl")

    return model

def predict(input):
    """
    Function that loads trained logistic regression model
    and predicts label of given review.
    Input: Vectorized user input
    Output: label (-1: bad, 0 : good)
    """
    print("PREDICTING!")

    model = load_model()

    #Make a prediction
    prediction = model.predict(input)

    return prediction
