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

PIPELINE_NAME = "MODEL VALIDATION PIPELINE"

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_validation_config = config_manager.get_model_validation_config()
        model_validator = ModelValidator(config=model_validation_config)

        # Determine next run number
        root_dir = model_validation_config.root_dir
        run_number = get_run_count_from_file(root_dir) + 1

        # Validate the model
        model_validator.validate(run_number)

        # Write the updated run number back to the file
        write_run_count_to_file(root_dir, run_number)

        logger.info("Model Validation Completed Successfully")

    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}")
        wandb.finish()
        sys.exit(1)