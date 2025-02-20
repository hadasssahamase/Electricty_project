from pathlib import Path
import motor.motor_asyncio  
import pandas as pd
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.config_entity.config_params import DataIngestionConfig


# Load the environment variables
load_dotenv()


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.X_test_transformed, self.y_test = self.load_and_prepare_data()


    def load_and_prepare_data(self) -> Tuple[Any, pd.Series]:

        try:
            data_manager = DataManager()
            X_test_transformed, y_test = data_manager.load_evaluation_data(
                self.config.test_features_path,
                self.config.test_targets_path,
            )

            logger.info(f"Loaded the test data successfully")

            return X_test_transformed, y_test
        
        except Exception as e:
            logger.error(f"Failed to load the test data")
            raise CustomException(e, sys)


    def load_model(self):
        try:
            model_path = self.config.model_path
            gb_model = joblib.load(model_path)
            logger.info(f"Loaded the pre-trained model from: {model_path}")
            return gb_model
        
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)

    def evaluate(self, run_number: int):
        try:
            # Load the model 
            gb_model = self.load_model()

            # Initialize the WandB run with dynamic run name 
            run_name = f"Evaluation {run_number}"
            run = wandb.init(
                project = self.config.project_name,
                name = run_name,
                config = {"random_state": self.config.random_state}
            )

            # Evaluate on testing data 
            y_test_pred = gb_model.predict(self.X_test_transformed)

            # Calculate the metrics 
            mae = mean_absolute_error(self.y_test, y_test_pred)
            mse = mean_squared_error(self.y_test, y_test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_test_pred)

            # Calculate the adjusted R-squared
            n = len(self.y_test)  # Number of samples
            p = self.X_test_transformed.shape[1]  # Number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            # MAPE (Mean Absolute Percent Error)
            def calculate_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = calculate_mape(self.y_test, y_test_pred)

            # Log the metrics to WandB 
            wandb.log({"Evaluation_mae": mae, "Evaluation_mse": mse, 
                       "Evaluation_rmse": rmse, "Evaluation_r2": r2, 
                       "Evaluation_adjusted_r2": adjusted_r2, "Evaluation_mape": mape})

            logger.info(f"Evaluation metrics logged")

            run.finish()    

        except Exception as e:
            logger.error(f"Failed to evaluate the model")
            raise CustomException(e, sys)