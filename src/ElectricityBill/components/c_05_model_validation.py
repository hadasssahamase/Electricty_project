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


class ModelValidator:
    def __init__(self, config: ModelValidationConfig) -> None:
        self.config = config
        self.X_val_transformed, self.y_val = self.load_and_prepare_data()

    def load_and_prepare_data(self) -> Tuple[Any, pd.Series]:
        """Loads and prepares the validation data."""
        try:
            data_manager = DataManager()
            X_val_transformed, y_val = data_manager.load_validation_data(
                self.config.val_features_path,
                self.config.val_targets_path,
            )
            logger.info("Validation data loaded and prepared successfully.")

            return X_val_transformed, y_val

        except Exception as e:
            logger.error(f"Error loading and preparing validation data: {str(e)}")
            raise CustomException(e, sys)

    def load_model(self):
        """Loads the pre-trained model."""
        try:
            model_path = self.config.model_path
            gb_model = joblib.load(model_path)
            logger.info(f"Loaded pre-trained model from: {model_path}")
            return gb_model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)

    def validate(self, run_number: int):
        try:

            # Load pre-trained model
            gb_model = self.load_model()

            # Initialize WandB run with a dynamic run name
            run_name = f"Validation {run_number}"
            run = wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={"random_state": self.config.random_state}
            )

            # Evaluate on validation set
            y_val_pred = gb_model.predict(self.X_val_transformed)

            # Calculate Metrics
            mae = mean_absolute_error(self.y_val, y_val_pred)
            mse = mean_squared_error(self.y_val, y_val_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_val, y_val_pred)

            # Calculate Adjusted R-squared
            n = len(self.y_val)  # Number of samples
            p = self.X_val_transformed.shape[1]  # Number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            # MAPE (Mean Absolute Percentage Error) - An important metric for regression
            def calculate_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = calculate_mape(self.y_val, y_val_pred)

            # Log Metrics to WandB
            wandb.log({
                "validation_mae": mae,
                "validation_mse": mse,
                "validation_rmse": rmse,
                "validation_r2": r2,
                "validation_adjusted_r2": adjusted_r2,
                "validation_mape": mape  # Log MAPE
            })

            logger.info(f"Validation metrics logged to WandB run: {run_name}")

            run.finish()

        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            raise CustomException(e, sys)


def get_run_count_from_file(root_dir: str, filename="run_count.txt") -> int:
    """
    Reads the current run count from a single file.
    If the file doesn't exist, returns 0.
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'r') as f:
            count = int(f.read().strip())
        return count
    except FileNotFoundError:
        return 0
    except ValueError:
        logger.warning("Run count file is corrupted. Resetting to 0.")
        return 0  # Handle case where the file contains non-integer data


def write_run_count_to_file(root_dir: str, count: int, filename="run_count.txt") -> None:
    """
    Writes the new run count to the single tracking file.
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(str(count))
    except Exception as e:
        logger.error(f"Error writing run count to file: {str(e)}")
