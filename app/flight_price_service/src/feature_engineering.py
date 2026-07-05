import os
import yaml
import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from shared.logger import logger
from shared.exception import CustomException


class FeatureEngineer:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            self.model_dir = self.config["models"]["save_dir"]
            os.makedirs(self.model_dir, exist_ok=True)

            logger.info("FeatureEngineer initialized")

        except Exception as e:
            raise CustomException(f"Config loading failed: {e}")

    # ----------------------------
    # Drop unnecessary columns
    # ----------------------------
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop(columns=["travelCode", "userCode"], errors="ignore")
            logger.info("Dropped unnecessary columns")
            return df

        except Exception as e:
            raise CustomException(f"Drop columns failed: {e}")

    # ----------------------------
    # Date feature engineering
    # ----------------------------
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["date"] = pd.to_datetime(df["date"])

            df["day"] = df["date"].dt.day
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year

            df.drop("date", axis=1, inplace=True)

            logger.info("Date features created")
            return df

        except Exception as e:
            raise CustomException(f"Date feature creation failed: {e}")

    # ----------------------------
    # OneHot Encoding
    # ----------------------------
    def encode_features(self, df: pd.DataFrame):
        try:
            categorical_cols = ["from", "to", "flightType", "agency"]

            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

            transformer = ColumnTransformer(
                transformers=[
                    ("cat", encoder, categorical_cols)
                ],
                remainder="passthrough"
            )

            X_encoded = transformer.fit_transform(df)

            # Save transformer
            joblib.dump(
                transformer,
                os.path.join(self.model_dir, "ohe_transformer.pkl")
            )

            logger.info("OneHotEncoder saved and features encoded")

            return X_encoded

        except Exception as e:
            raise CustomException(f"Encoding failed: {e}")

    # ----------------------------
    # Save feature columns (optional)
    # ----------------------------
    def save_feature_columns(self, columns):
        try:
            joblib.dump(
                columns,
                os.path.join(self.model_dir, "feature_columns.pkl")
            )

            logger.info("Feature columns saved")

        except Exception as e:
            raise CustomException(f"Feature saving failed: {e}")