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

from src.ElectricityBill.logger import logger  
from src.ElectricityBill.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.ElectricityBill.pipelines.pip_02_data_validation import DataValidationPipeline
from src.ElectricityBill.pipelines.pip_03_data_transformation import DataTransformationPipeline
from src.ElectricityBill.pipelines.pip_04_model_trainer import ModelTrainerPipeline
from src.ElectricityBill.pipelines.pip_05_model_evaluation import ModelEvaluationPipelin

COMPONENT_01_NAME = "DATA INGESTION COMPONENT"
try: 
    logger.info(f"# ====================== {COMPONENT_01_NAME} Started! ============================== #")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f"# ====================== {COMPONENT_01_NAME} Terminated Successfully! ===============##\n\nx******************x")
except Exception as e:
    logger.exception(e)
    raise e

COMPONENT_02_NAME = "DATA VALIDATION COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_02_NAME} Started! ================================= #")
    data_validation_pipeline = DataValidationPipeline()
    data_validation_pipeline.run()
    logger.info(f"## ======================== {COMPONENT_02_NAME} Terminated Successfully!=============== ##\n\nx************************x")

except Exception as e:
    logger.exception(e)
    raise e

COMPONENT_03_NAME = "DATA TRANSFORMATION COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_03_NAME} Started! ================================= #")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.run()
    logger.info(f"## ======================== {COMPONENT_03_NAME} Terminated Successfully!=================== ##\n\nx*********************x")

except Exception as e:
    logger.exception(e)
    raise e

COMPONENT_04_NAME = "MODEL TRAINER COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_04_NAME} Started! ================================= #")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.run()
    logger.info(f"## ========================  {COMPONENT_04_NAME} Terminated Successfully!===================== ##\n\nx******************x")

except Exception as e:
    logger.exception(e)
    raise e


COMPONENT_05_NAME = "MODEL EVALUATION COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_05_NAME} Started! ================================= #")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.run()
    logger.info(f"## ======================== {COMPONENT_05_NAME} Terminated Successfully!======================= ##\n\nx******************x")
except Exception as e:
    logger.exception(e)
    raise e