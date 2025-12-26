import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pandas as pd

MODEL_DIR = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\model\\flight_prediction"

df = pd.read_csv("H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\data\\flight_prediction\\processed_flights.csv")

X = df.drop("price", axis=1)
y = df["price"]

joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Start MLflow run
with mlflow.start_run():

    # Model parameters
    params = {
        'n_estimators': 500,
        'learning_rate': 0.08,
        'max_depth': 5,
        'subsample': 0.88
    }

    # Log parameters
    mlflow.log_params(params)


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.88,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, f"{MODEL_DIR}/xgb_regressor.pkl")

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics
mlflow.log_metric('mse', mse)
mlflow.log_metric('r2', r2)

# Log model
mlflow.sklearn.log_model(model, artifact_path='xgb_model')

print(f'MSE: {mse}, R2: {r2}')