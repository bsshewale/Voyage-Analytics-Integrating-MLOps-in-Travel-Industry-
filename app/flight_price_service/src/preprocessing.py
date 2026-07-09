import pandas as pd
from shared.logger import logger
from shared.exception import CustomException


class DataPreprocessor:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning: duplicates, nulls, type fixes
        """
        try:
            logger.info("Starting data preprocessing...")

            # 1. Remove duplicates
            df = df.drop_duplicates()
            logger.info(f"Duplicates removed. Shape: {df.shape}")

            # 2. Check missing values
            missing = df.isnull().sum().sum()
            logger.info(f"Total missing values: {missing}")

            # 3. Type conversion (IMPORTANT from your notebook)
            df["price"] = df["price"].astype(int)

            # Convert date safely
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Drop rows where date failed conversion
            df = df.dropna(subset=["date"])

            logger.info("Type conversion completed")

            return df

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise CustomException(f"Preprocessing Failed: {e}")

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional validation step
        """
        try:
            required_columns = [
                "travelCode",
                "userCode",
                "from",
                "to",
                "flightType",
                "price",
                "time",
                "distance",
                "agency",
                "date",
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            logger.info("Data validation passed")

            return df

        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise CustomException(f"Validation Failed: {e}")