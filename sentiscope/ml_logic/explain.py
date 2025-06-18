#DL explaining
from lime.lime_text import LimeTextExplainer

#Required imports
import tensorflow as tf

#Preprocessing and prediction
from sentiscope.ml_logic.model import predict
from sentiscope.ml_logic.preprocessor import preprocess_ml, preprocess_dl

#Variables for Lime
class_names = ["Negative", "Positive"]


def create_predict_fn(model, tokenizer):
    """
    Function to create a prediction function which can be called by the API
    directly
    """

    @tf.function
    def predict_logits(tokens):
        """
        Optimized function to create a prediction of the trained bert model

        Input: TF Tensor
        Output: TF prediction logits
        """
        return model(tokens, training=False).logits

    def predict_proba(review):
        """
        Function which preprocesses user input, calls predict function and turns
        the returned values to probabilities of the review belonging to either
        the positive or negative class using softmax.

        Input: review string
        Output: numpy array of 2 probabilities
        """

        #Preprocess input and fetch prediction
        tokenized = preprocess_dl(review,tokenizer)
        logits = predict_logits(tokenized)

        #Calculate probas with softmax
        probs = tf.nn.softmax(logits, axis=1)
        return probs.numpy()

    #Return function so API can call it
    return predict_proba

def explain_with_lime(text, predict_proba_fn, num_features=10):
    """
    Function which tracks which parts of a review had the most positive or
    negative influence on the prediction of the model
    """

    #Calculate contributions of each word of the review
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba_fn,
        num_features=num_features,
        labels=[1],
        num_samples=300
    )

    # Sort contributions and extract the words which have the most impact
    # positively and negatively
    contribs = dict(exp.as_list(label=1))
    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1])
    top_negative = [w for w, score in sorted_contribs if score < 0][:2]
    top_positive = \
        [w for w, score in reversed(sorted_contribs) if score > 0][:2]

    #Return sentiment and calculated contributions + top positive/negative words
    sentiment = "Positive" if \
        predict_proba_fn([text])[0][1] > 0.5 else "Negative"
    return {
        "sentiment": sentiment,
        "contributions": contribs,
        "top_positive": top_positive,
        "top_negative": top_negative
    }


def explain_ml(review,model, pipeline):
    #Preprocess the review
    X_pred = preprocess_ml(review,pipeline)

    #Predict the sentiment of the review
    prediction = predict(X_pred, model)

    #Extract vectorizer for visualization data
    vectorizer = pipeline['vectorizer']

    #Extract data from model and vectorizer
    coefs = model.coef_[0]
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
