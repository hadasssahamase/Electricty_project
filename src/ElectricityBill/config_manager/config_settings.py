import sys
sys.path.append('C:/Users/ADMIN/OneDrive/Desktop/hadadata/Electricity_project/Electricty_project')


from dataclasses import dataclass
from pathlib import Path
import motor.motor_asyncio  
import pandas as pd
import numpy as np
import os
import json
import time
import asyncio  
from datetime import datetime
from dotenv import load_dotenv
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill.logger import logger
from src.ElectricityBill.constants import *
from src.ElectricityBill.utils.commons import read_yaml, create_directories
from src.ElectricityBill.config_entity.config_params import *
# Load the environment variables
load_dotenv()

class ConfigurationManager:
    def __init__(self, 
                 data_ingestion_config: Path = DATA_INGESTION_CONFIG_FILEPATH,
                 data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
                 schema_config: Path = SCHEMA_CONFIG_FILEPATH,
                 data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH,
                 model_training_config = MODEL_TRAINER_CONFIG_FILEPATH,
                 model_params_config = PARAMS_CONFIG_FILEPATH,
                 hyperparameter_config: Path = HYPERPARAMETER_SEARCH_CONFIG_FILEPATH,
                 model_validation_config_path: Path = MODEL_VALIDATION_CONFIG_FILEPATH,
                 model_evaluation_config_path: Path = MODEL_EVALUATION_CONFIG_FILEPATH
                 )-> None:
        try:
            logger.info(f"Initializing ConfigurationManager with config files")
            
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config)
            self.preprocessing_config = read_yaml(data_preprocessing_config) 
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)
            self.wandb_config = read_yaml(hyperparameter_config)
            self.model_validation_config = read_yaml(model_validation_config_path)
            self.model_evaluation_config = read_yaml(model_evaluation_config_path)
            
            
            create_directories([self.ingestion_config.artifacts_root])
            create_directories([self.data_val_config.artifacts_root])
            create_directories([self.preprocessing_config.artifacts_root])
            create_directories([self.training_config.artifacts_root])
            create_directories([self.model_evaluation_config.artifacts_root])
            create_directories([self.model_validation_config.artifacts_root]) 
            
            logger.info("Configuration directories created successfully.")

           
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            logger.error(f"Error creating directories")
            raise CustomException(e, sys)
        
    


# Data ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
    
    def get_user_name(self):
        try:
            return self.ingestion_config['data_ingestion'].get('get_user_name', 'DefaultUser')
        except Exception as e:
            logger.error(f"Error getting user name from config: {e}")
            raise CustomException(e, sys)
        
# Data Validation configuration

    def get_data_validation_config(self) -> Configuration:
        try:
            data_valid_config = self.data_val_config.data_validation 
            schema_dict = self._process_schema()
            profile_report_path = os.path.join(data_valid_config.root_dir, "data_profile_report.html") # Define profile report path
            create_directories([Path(data_valid_config.root_dir)]) 
            logger.info(f"Data Validation Config Loaded") 

            return Configuration(DataValidationConfig(
                root_dir = Path(data_valid_config.root_dir), 
                val_status = data_valid_config.val_status, 
                data_dir = Path(data_valid_config.data_dir), 
                all_schema = schema_dict,
                critical_columns = data_valid_config.critical_columns,
                profile_report_path=profile_report_path  #Store path to config
            ))
        except Exception as e: 
            logger.exception(f"Error getting data validation configuration: {str(e)}") 
            raise CustomException(e, sys)

    def _process_schema(self) -> Dict[str, str]:
        schema_columns = self.schema.get("columns", {})
        target_column = self.schema.get("target_column", [])
        schema_dict = {col['name']: col['type'] for col in schema_columns}
        schema_dict.update({col['name']: col['type'] for col in target_column})
        return schema_dict   
    
# Data Transformation     

    def get_data_transformation_config(self) -> DataTransformationConfig:
        logger.info("Getting data transformation configuration")

        transformation_config = self.preprocessing_config.data_transformation
        create_directories([transformation_config.root_dir])

        return DataTransformationConfig(
            root_dir = Path(transformation_config.root_dir),
            data_path = Path(transformation_config.data_path),
            numerical_cols = transformation_config.numerical_cols,
            categorical_cols = transformation_config.categorical_cols,
            target_col = transformation_config.target_col,
            random_state = transformation_config.random_state
        )
        
        
 # Model trainer
    def get_model_training_config(self) -> ModelTrainerConfig:
        try:
            trainer_config = self.training_config['model_trainer']  # Dictionary access here
            model_params = self.model_params_config['GradientBoostingRegressor'] #Added key to fetch model params

            return ModelTrainerConfig(
                root_dir=Path(trainer_config['root_dir']),  # Dictionary access here
                train_features_path=Path(trainer_config['train_features_path']),  # Dictionary access here
                train_targets_path=Path(trainer_config['train_targets_path']),  # Dictionary access here
                model_name=trainer_config['model_name'],  # Dictionary access here
                model_params=model_params,
                project_name=trainer_config['project_name'],  # Dictionary access here
                random_state=trainer_config['random_state'],  # Dictionary access here
                number_of_splits=int(trainer_config['number_of_splits'])  # Dictionary access here
            )

        except Exception as e:
            logger.error(f"Error in getting model training configuration: {str(e)}")
            raise CustomException(e, sys) # Pass the exception to CustomException
       
       
    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config["model_trainer"]
            model_params = self.model_params_config["GradientBoostingRegressor"]  

            # Creates all necessary directories
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params,
                project_name=trainer_config.project_name,
                val_features_path=Path(trainer_config.val_features_path),
                val_targets_path=Path(trainer_config.val_targets_path),
            )
        except Exception as e:
            logger.error(f"Error getting model training config: {str(e)}")
            raise CustomException(e, sys)
     
# Model Validation
    def get_model_validation_config(self) -> ModelValidationConfig:
        logger.info("Getting model validation configuration")
        try:
            model_val_config = self.model_validation_config['model_validation']  # Correct key name
            # Create directories
            create_directories([model_val_config['root_dir']])

            return ModelValidationConfig(
                root_dir=Path(model_val_config['root_dir']),
                val_features_path=Path(model_val_config['val_features_path']),
                val_targets_path=Path(model_val_config['val_targets_path']),
                model_path=Path(model_val_config['model_path']),
                project_name=model_val_config['project_name'],
                random_state=int(model_val_config['random_state'])
            )

        except Exception as e:
            logger.error(f"Error in getting model validation configuration: {str(e)}")
            raise CustomException(e, sys)


# Model Evaluation

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