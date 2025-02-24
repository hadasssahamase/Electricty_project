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
from src.ElectricityBill.constants import DATA_VALIDATION_CONFIG_FILEPATH
from src.ElectricityBill.utils.commons import read_yaml, create_directories
from src.ElectricityBill.config_manager.config_settings import *
from src.ElectricityBill.pipelines.pip_02_data_validation import DataValidationPipeline
from src.ElectricityBill.components.c_02_data_validation import *

PIPELINE_NAME = "DATA VALIDATION PIPELINE"
class DataValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()

            if validation_status:
                print("Data validation successful!")
            else:
                print("Data validation failed.")

        except Exception as e:
            print(f"Error: {e}")
         
    
if __name__ == "__main__":
    try:
        logger.info (f"------------> starting {PIPELINE_NAME} pipeline ------------->")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logger.info(f"------------> {PIPELINE_NAME} pipeline completed successfully ------------->")

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME} pipeline: {e}")
        raise CustomException(e, sys)    