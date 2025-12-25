import joblib
from .config import MODEL_DIR

model = joblib.load(f"{MODEL_DIR}/xgb_regressor.pkl")
feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")

def predict(X):
    X = X[feature_columns]
    return model.predict(X)
