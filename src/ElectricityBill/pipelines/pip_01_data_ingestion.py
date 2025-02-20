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
from src.ElectricityBill.config_manager.config_settings import *

PIPELINE_NAME = "DATA INGESTION PIPELINE"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            user_name = config_manager.get_user_name()
            data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
            asyncio.run(data_ingestion.import_data_from_mongodb())  # Updated to run async function
            logger.info("Data ingestion process completed successfully.")
        except CustomException as e:
            logger.error(f"Error during data ingestion: {e}")
            logger.info("Data ingestion process failed.")


if __name__ == "__main__":
    try:
        logger.info (f"------------> starting {PIPELINE_NAME} pipeline ------------->")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f"------------> {PIPELINE_NAME} pipeline completed successfully ------------->")

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME} pipeline: {e}")
        raise CustomException(e, sys)