import os
import pandas as pd
import yaml
from shared.logger import logger
from shared.exception import CustomException


class DataIngestion:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            self.raw_data_path = self.config["data"]["raw_data_path"]

        except Exception as e:
            raise CustomException(f"Error loading config: {e}")

    def load_data(self):
        """
        Load raw dataset from CSV
        """
        try:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"File not found: {self.raw_data_path}")

            df = pd.read_csv(self.raw_data_path)

            logger.info(f"Data loaded successfully with shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(f"Data Ingestion Failed: {e}")


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    print(df.head())