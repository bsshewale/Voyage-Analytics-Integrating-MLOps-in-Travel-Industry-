import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import joblib
import os

# Load data
DATA_PATH = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\data\\gender_classifier"
MODEL_DIR = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\model\\gender_classifier"
df = pd.read_csv(DATA_PATH + "\\users.csv")

# Drop unnecessary columns
df = df.drop(columns=["code", "company"])

# Keep only male/female for training
# Encode target
df["gender"] = (
    df["gender"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["gender"] = df["gender"].map({
    "male": 0,
    "female": 1,
    "none": 2
})

# TF-IDF on names (character-level)
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1
)

X_name = tfidf.fit_transform(df["name"])
X_age = df[["age"]].values

X = hstack([X_name, X_age])
y = df["gender"]

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save artifacts
joblib.dump(model, MODEL_DIR + "\\model.joblib")
joblib.dump(tfidf, MODEL_DIR + "\\tfidf.joblib")

# Save processed data (optional)
df.to_csv(MODEL_DIR + "\\processed_users.csv", index=False)