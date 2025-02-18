import sys
sys.path.append('C:/Users/ADMIN/OneDrive/Desktop/hadadata/Electricity_project/Electricty_project')



from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import json
import joblib
import wandb

from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import *


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_features_path: Path
    test_targets_path: Path
    model_path: Path
    project_name: str
    random_state: int


class ConfigurationManager:
    def __init__(self, 
                 model_evaluation_config_path: Path = MODEL_EVALUATION_CONFIG_FILEPATH) -> None:
        try:
            self.model_evaluation_config = read_yaml(model_evaluation_config_path)
            create_directories([self.model_evaluation_config['artifacts_root']])
        except Exception as e:
            logger.error(f"Failed to load model evaluation configuration files: {str(e)}")
            raise CustomException (e, sys)
        

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info(f"Getting the model evaluation configuration")

        try:
            model_eval = self.model_evaluation_config['model_evaluation'] #Change with square brackets
            create_directories([model_eval['root_dir']]) #Change with square brackets

            return ModelEvaluationConfig(
                root_dir = model_eval['root_dir'],  #Change with square brackets
                test_features_path = Path(model_eval['test_features_path']),  #Change with square brackets
                test_targets_path = Path(model_eval['test_targets_path']),  #Change with square brackets
                model_path = Path(model_eval['model_path']),  #Change with square brackets
                project_name = model_eval['project_name'],  #Change with square brackets
                random_state = model_eval['random_state'],  #Change with square brackets
            )

        except Exception as e:
            logger.error(f"Failed to load the model evaluation configuration: {str(e)}")
            raise CustomException (e, sys)

class DataManager:
    @staticmethod
    def load_data(feature_path: Path, target_path: Path) -> Tuple[Any, pd.Series]:
        """Loads the feature and target data."""
        try:
            logger.info(f"Loading feature data from {feature_path}")
            X_transformed = joblib.load(feature_path)
            logger.info(f"Loading target data from {target_path}")
            y = pd.read_parquet(target_path.as_posix()).squeeze()
            
            return X_transformed, y

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)

    
    @staticmethod
    def load_evaluation_data(test_feature_path: Path, test_target_path: Path) -> Tuple[Any, pd.Series]:
        """Loads the evaluation data from the specified paths."""
        try:
            X_test_transformed, y_test = DataManager.load_data(test_feature_path, test_target_path)
            logger.info("Successfully loaded evaluation data.")
            return X_test_transformed, y_test

        except Exception as e:
            logger.error(f"Error loading evaluation data: {str(e)}")
            raise CustomException(e, sys)

        
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

def get_run_count(root_dir: str, filename:str = "eval_run_count.txt") -> int:
    """
    Reads the current evaluation run count from a single file.
    Returns 0 if file does not exists
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'r') as f:
            count = int(f.read().strip())
            return count
    except FileNotFoundError:
        return 0

    except ValueError:
        logger.warning(f"Failed to read the evaluation run count")
        return 0

def write_run_count(root_dir: str, count:int, filename:str = "eval_run_count.txt") -> None:
    """
    Writes the current evaluation run count to a single file
    """
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(str(count))
    except Exception as e:
        logger.error(f"Failed to write the evaluation run count")
        raise CustomException(e, sys)
    

if __name__ == '__main__':
    try:
        # Initialize the configuration manager
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluator(config = model_evaluation_config)

        ## Determine next run number
        root_dir = model_evaluation_config.root_dir

        run_number = get_run_count(root_dir) + 1

        # Validate the model
        model_evaluator.evaluate(run_number)

        # Write the updated run number back to the file
        write_run_count(root_dir, run_number)

        logger.info("Model Evaluation Completed Successfully")

    except CustomException as ce:
        logger.error(f"Error in model evaluation")
        wandb.finish()
        sys.exit(1)

# import sys
# from pathlib import Path
# from dataclasses import dataclass
# from typing import Tuple
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import os
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import wandb
# from src.ElectricityBill.exception import CustomException
# from src.ElectricityBill.logger import logger
# from src.ElectricityBill.constants import *
# from src.ElectricityBill.utils.commons import *
# wandb.require("core")
# @dataclass
# class ModelValidationConfig:
#     root_dir: Path
#     test_feature_path: Path
#     test_targets_path: Path
#     model_path: Path
#     validation_scores_path: Path
#     project_name: str
# class ConfigurationManager:
#     def __init__(self, model_validation_config: str = MODEL_VALIDATION_CONFIG_FILEPATH):
#         self.validation_config = read_yaml(model_validation_config)
#         artifacts_root = self.validation_config.artifacts_root
#         create_directories([artifacts_root])
#     def get_model_validation_config(self) -> ModelValidationConfig:
#         logger.info("Getting model validation configuration")
#         val_config = self.validation_config.model_validation
#         create_directories([val_config.root_dir])
#         return ModelValidationConfig(
#             root_dir=Path(val_config.root_dir),
#             test_feature_path=Path(val_config.test_feature_path),
#             test_targets_path=Path(val_config.test_targets_path),
#             model_path=Path(val_config.model_path),
#             validation_scores_path=Path(val_config.validation_scores_path),
#             project_name = val_config.project_name
#         )
# class ModelValidation:
#     def __init__(self, config: ModelValidationConfig):
#         self.config = config
#         self.model = None  # Initialize model attribute
#         self.X_test = None
#         self.y_test = None
#         self.run = None
#     def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
#         """
#         Logs the start of model validation data loading and retrieves
#         paths for test features and test targets from the configuration.
#         """
#         logger.info("Loading model validation data")
#         try:
#             test_features = self.config.test_feature_path
#             test_targets = self.config.test_targets_path
#             X_test = joblib.load(test_features)
#             y_test = pd.read_parquet(test_targets)
#             # Convert y_test to a Series if it's a DataFrame
#             if isinstance(y_test, pd.DataFrame):
#                 y_test = y_test.squeeze()
#             logger.info("Model validation data loaded successfully")
#             return X_test, y_test
#         except Exception as e:
#             logger.error(f"Error loading data: {e}")
#             raise CustomException(f"Error loading data: {e}")
#     def load_model(self) -> object:
#         logger.info("Loading model")
#         try:
#             model = joblib.load(self.config.model_path)
#             logger.info("Model loaded successfully")
#             return model
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             raise CustomException(f"Error loading model: {e}")
#     def validate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
#         """
#         Validates the loaded model on the test data and logs metrics.
#         """
#         logger.info("Validating the model on the test data")
#         try:
#             # Make predictions
#             y_pred = self.model.predict(X_test)
#             # Calculate metrics
#             mae = mean_absolute_error(y_test, y_pred)
#             mse = mean_squared_error(y_test, y_pred)
#             rmse = np.sqrt(mse)
#             r2 = r2_score(y_test, y_pred)
#             # Calculate Adjusted R-squared
#             n = X_test.shape[0]  # Number of samples
#             p = X_test.shape[1]  # Number of features
#             adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#             mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#             # Log metrics to W&B
#             metrics = {
#                 "MAE": mae,
#                 "MSE": mse,
#                 "RMSE": rmse,
#                 "R2": r2,
#                 "Adjusted R2": adjusted_r2,
#                 "MAPE": mape
#             }
#             logger.info(f"Validation metrics: {metrics}")
#             # Save metrics to JSON file
#             with open(self.config.validation_scores_path, 'w') as f:
#                 json.dump(metrics, f, indent=4)  # Add indent for readability
#             # Log metrics to W&B if run is initialized
#             if self.run:
#                 self.run.log(metrics)
#         except Exception as e:
#             logger.error(f"Error during model validation: {e}")
#             raise CustomException(f"Error during model validation: {e}", error_details=str(e))
#     def run_validation(self):
#         """
#         Runs the complete model validation process.
#         """
#         try:
#             # Initialize W&B run
#             self.run = wandb.init(project=self.config.project_name, config=self.config.__dict__)
#             X_test, y_test = self.load_data()
#             model = self.load_model()
#             self.model = model
#             self.validate_model(X_test, y_test)  # Pass data and model to the validation function
#         except Exception as e:
#             logger.error(f"Error during model validation: {e}")
#             raise CustomException(f"Error during model validation: {e}", error_details=str(e))
#         finally:
#             if self.run:
#                 self.run.finish()
# if __name__ == "__main__":
#     try:
#         config = ConfigurationManager()
#         model_validation_config = config.get_model_validation_config()
#         model_validation = ModelValidation(config=model_validation_config)
#         model_validation.run_validation()
#     except Exception as e:
#         logger.error(f"Error during the model validation main process: {e}")
#         raise CustomException(f"Error during the model validation main process: {e}")