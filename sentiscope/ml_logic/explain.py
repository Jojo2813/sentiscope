from lime.lime_text import LimeTextExplainer
import tensorflow as tf
import numpy as np

class_names = ["Negative", "Positive"]

def create_predict_fn(model, tokenizer):

    @tf.function
    def predict_logits(inputs):
        return model(inputs, training=False).logits

    def predict_proba(texts):
        tokenized = tokenizer(
            texts,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=400
        )
        logits = predict_logits(tokenized)
        probs = tf.nn.softmax(logits, axis=1)
        return probs.numpy()

    return predict_proba

def explain_with_lime(text, predict_proba_fn, num_features=10):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba_fn,
        num_features=num_features,
        labels=[1],
        num_samples=300
    )
    contribs = dict(exp.as_list(label=1))
    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1])
    top_negative = [w for w, score in sorted_contribs if score < 0][:2]
    top_positive = [w for w, score in reversed(sorted_contribs) if score > 0][:2]
    sentiment = "Positive" if predict_proba_fn([text])[0][1] > 0.5 else "Negative"
    return {
        "sentiment": sentiment,
        "contributions": contribs,
        "top_positive": top_positive,
        "top_negative": top_negative
    }
