import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from .data_loader import load_data
from .preprocessing import fit_encoders
from .config import DATA_DIR, MODEL_DIR
import pandas as pd

df = load_data(f"{DATA_DIR}/flights.csv")

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df.drop(["date", "travelCode", "userCode"], axis=1, inplace=True)

df = fit_encoders(df)

X = df.drop("price", axis=1)
y = df["price"]

joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.88,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, f"{MODEL_DIR}/xgb_regressor.pkl")
