import os
import joblib
import numpy as np
import pandas as pd
import yaml

from shared.logger import logger
from shared.exception import CustomException


class FlightPricePredictor:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            self.model_path = os.path.join(
                self.config["models"]["save_dir"],
                "xgb_regressor.pkl"
            )

            self.encoder_path = os.path.join(
                self.config["models"]["save_dir"],
                "ohe_transformer.pkl"
            )

            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)

            logger.info("Model and encoder loaded successfully")

        except Exception as e:
            raise CustomException(f"Predictor init failed: {e}")

    def preprocess_input(self, df: pd.DataFrame):
        try:
            # Drop IDs if present
            df = df.drop(columns=["travelCode", "userCode"], errors="ignore")

            # Date features
            df["date"] = pd.to_datetime(df["date"])
            df["day"] = df["date"].dt.day
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year
            df = df.drop("date", axis=1)

            # Separate categorical columns
            cat_cols = ["from", "to", "flightType", "agency"]

            # Transform using saved encoder
            encoded = self.encoder.transform(df)

            return encoded

        except Exception as e:
            raise CustomException(f"Preprocessing failed: {e}")

    def predict(self, input_data: dict):
        try:
            logger.info("Prediction started")

            # Convert input dict → DataFrame
            df = pd.DataFrame([input_data])

            # Preprocess
            X = self.preprocess_input(df)

            # Predict
            prediction = self.model.predict(X)

            logger.info(f"Prediction completed: {prediction[0]}")

            return float(prediction[0])

        except Exception as e:
            raise CustomException(f"Prediction failed: {e}")


# --------------------------
# Example test run
# --------------------------
if __name__ == "__main__":
    predictor = FlightPricePredictor()

    sample_input = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "date": "2019-09-26"
    }

    result = predictor.predict(sample_input)
    print("Predicted Price:", result)