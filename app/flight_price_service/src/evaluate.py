import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shared.logger import logger
from shared.exception import CustomException


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate regression model performance
        """
        try:
            logger.info("Starting model evaluation...")

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            results = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            }

            logger.info(f"Evaluation Results: {results}")

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise CustomException(f"Model Evaluation Failed: {e}")