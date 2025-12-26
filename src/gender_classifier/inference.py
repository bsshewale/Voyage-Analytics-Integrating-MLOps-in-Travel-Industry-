import numpy as np
import joblib
from scipy.sparse import hstack

model = joblib.load("model/gender_classifier/model.joblib")
tfidf = joblib.load("model/gender_classifier/tfidf.joblib")

label_reverse = {0: "male", 1: "female", 2: "none"}

def predict_gender(name: str, age: int):
    name_vec = tfidf.transform([name.strip()])
    age_vec = np.array([[age]])
    X_input = hstack([name_vec, age_vec])

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    return {
        "gender": label_reverse[pred],
        "confidence": round(float(max(proba)) * 100, 2)
    }
