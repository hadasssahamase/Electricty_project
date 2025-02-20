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
from src.ElectricityBill.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.ElectricityBill.utils.commons import read_yaml, create_directories


PIPELINE_NAME = "MODEL TRAINER PIPELINE"
class DataModelTrainerPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config_manager = ConfigurationManager()
            model_training_config = config_manager.get_model_training_config()
            model_trainer = ModelTrainer(config=model_training_config)

        # Train the model
            model = model_trainer.train() #
            logger.info("Model Training Completed Successfully")

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
        wandb.finish()
        sys.exit(1)
        
        
if __name__ == "__main__":
    try:
        logger.info (f"------------> starting {PIPELINE_NAME} pipeline ------------->")
        data_modeltrainer_pipeline = DataModelTrainerPipeline()
        data_modeltrainer_pipeline.run()
        logger.info(f"------------> {PIPELINE_NAME} pipeline completed successfully ------------->")

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME} pipeline: {e}")
        raise CustomException(e, sys)              