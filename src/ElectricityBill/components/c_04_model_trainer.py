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


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        self.X_train_transformed, self.y_train, self.X_val_transformed, self.y_val = self.load_and_prepare_data()

    def load_and_prepare_data(self) -> Tuple[Any, pd.Series, Any, pd.Series]:
        """Loads and prepares the training and validation data."""
        try:
            data_manager = DataManager()
            X_train_transformed, y_train, X_val_transformed, y_val = data_manager.load_all_data(
                self.config.train_features_path,
                self.config.train_targets_path,
            )
            logger.info("Data loaded and prepared successfully within ModelTrainer.")

            return X_train_transformed, y_train, X_val_transformed, y_val

        except Exception as e:
            logger.error(f"Error loading and preparing data: {str(e)}")
            raise CustomException(e, sys)

    def train(self): #No parameters needed here
        try:
            if not self.config.model_params:
                raise ValueError("Model parameters not provided.")

            run = wandb.init(
                project=self.config.project_name,
                config={**self.config.model_params, "random_state": self.config.random_state}
            )
            gb_model = GradientBoostingRegressor(**self.config.model_params,
                                                 random_state=self.config.random_state)

            # Perform K-Fold Cross validation
            kf = KFold(n_splits=self.config.number_of_splits, shuffle=True, random_state=self.config.random_state)

            cv_scores = cross_val_score(
                gb_model, self.X_train_transformed, self.y_train, cv=kf,
                scoring="neg_root_mean_squared_error"
            )  # Use 'neg_root_mean_squared_error'

            cv_rmse_scores = -cv_scores  # Convert back to positive RMSE
            mean_cv_rmse = np.mean(cv_rmse_scores)
            std_cv_rmse = np.std(cv_rmse_scores)

            logger.info(f"K-Fold Cross-validation RMSE scores: {cv_rmse_scores}")
            logger.info(f"Mean K-Fold cross-validation RMSE: {mean_cv_rmse}")
            logger.info(f"Standard deviation of K-Fold cross-validation RMSE: {std_cv_rmse}")

            wandb.log({"mean_cv_rmse": mean_cv_rmse, "std_cv_rmse": std_cv_rmse})

            # Fit the model on the entire training set AFTER cross-validation
            gb_model.fit(self.X_train_transformed, self.y_train)

            # Save the model, it's now trained on full data
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(gb_model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            run.finish()

            return gb_model
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)