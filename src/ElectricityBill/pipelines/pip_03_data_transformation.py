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

PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()

            data_transformation = DataTransformation(config=data_transformation_config)

            X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_split()

            preprocessor_obj, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test = data_transformation.initiate_data_transformation(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        except CustomException as e:
            logger.error(f"Error during data transformation: {e}")
        sys.exit(1)
        
        
        
if __name__ == "__main__":
    try:
        logger.info (f"------------> starting {PIPELINE_NAME} pipeline ------------->")
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.run()
        logger.info(f"------------> {PIPELINE_NAME} pipeline completed successfully ------------->")

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME} pipeline: {e}")
        raise CustomException(e, sys)            