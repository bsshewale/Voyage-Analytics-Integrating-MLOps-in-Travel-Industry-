import os
import joblib
import numpy as np
import yaml

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from shared.logger import logger
from shared.exception import CustomException

from src.feature_engineering import FeatureEngineer


class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            self.model_dir = self.config["models"]["save_dir"]
            self.test_size = self.config["training"]["test_size"]
            self.random_state = self.config["training"]["random_state"]

            os.makedirs(self.model_dir, exist_ok=True)

            logger.info("ModelTrainer initialized")

        except Exception as e:
            raise CustomException(f"Config load failed: {e}")

    def load_data(self):
        try:
            data_path = self.config["data"]["raw_data_path"]
            import pandas as pd
            df = pd.read_csv(data_path)
            return df

        except Exception as e:
            raise CustomException(f"Data loading failed: {e}")

    def train(self):
        try:
            logger.info("Training started")

            # -------------------------
            # Load data
            # -------------------------
            df = self.load_data()

            # -------------------------
            # Feature Engineering
            # -------------------------
            engineer = FeatureEngineer()

            df = engineer.drop_columns(df)
            df = engineer.create_date_features(df)

            # Split target
            y = df["price"]
            X = df.drop("price", axis=1)

            # Encode features
            X = engineer.encode_features(X)

            # -------------------------
            # Train-test split
            # -------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # -------------------------
            # Models
            # -------------------------
            models = {
                "DecisionTree": DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_leaf=50,
                    random_state=42
                ),

                "RandomForest": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=13,
                    n_jobs=-1,
                    random_state=42
                ),

                "XGBoost": XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.08,
                    max_depth=5,
                    subsample=0.88,
                    reg_alpha=0.0007,
                    reg_lambda=3.4
                )
            }

            best_model = None
            best_score = -np.inf
            best_name = ""

            # -------------------------
            # Train & Evaluate
            # -------------------------
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                logger.info(f"{name} -> MSE: {mse}, R2: {r2}")

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_name = name

            logger.info(f"Best Model: {best_name} | R2: {best_score}")

            # -------------------------
            # Save model
            # -------------------------
            model_path = os.path.join(self.model_dir, "xgb_regressor.pkl")
            joblib.dump(best_model, model_path)

            logger.info(f"Model saved at {model_path}")

            return best_model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CustomException(f"Training failed: {e}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()