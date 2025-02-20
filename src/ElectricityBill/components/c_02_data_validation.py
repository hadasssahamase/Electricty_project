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


class DataValidation:
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = logging.getLogger(__name__)  
        self.logger.info(f"Data Validation initialized")  
        self.logger.debug(f"Data validation config: {self.config}") 
        self.validators = [
            ColumnValidator(config),
            DataTypeValidator(config),
            MissingValueValidator(config),
            ConstraintValidator(config)
        ]
        
    def check_cardinality(self, data):
        """"Check  and drop columns with unique values"""

        unique_counts = data.nunique()
        drop_columns = [col for col in data.columns if unique_counts[col] == len(data)]
        if drop_columns: 
            logger.warning(f"Dropping columns with unique values: {drop_columns}")
    
        try:
            data.drop(columns=drop_columns, inplace=True)
            logger.debug(f"Dropped columns with unique values: {drop_columns}")
        except Exception as e:
            logger.error(f"Error occurred while dropping columns: {e}")
        
        return data
    
    def _generate_profile_report(self, data: pd.DataFrame) -> Dict:
        """
        Generates a pandas profile report and returns the path to the HTML report and also return the description.
        
        Args:
            data (pd.DataFrame): The DataFrame to profile
            
        Returns:
            Dict: the profile report and description.
        """
        try:
            report_path = Path(self.config.get_config().profile_report_path) #Get from config now
            
            profile = ProfileReport(data, title="Data Profiling Report")
            profile.to_file(str(report_path))
            description = profile.get_description() # Get the description
            
            self.logger.info(f"Profile report generated at: file://{report_path.absolute()}")
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating or saving pandas profile report: {e}")
            raise CustomException(e, sys)
        
    # And update the validate_data method to use the returned path:
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the data and return a status, save the metadata of the validation,
        and save the data if validation is passed
        """
        validation_results = {}
        overall_status = True
        profile_report = self._generate_profile_report(data) # Get report
        for validator in self.validators:
            result = validator.validate(data, profile_report)  #Pass the report
            validation_results[validator.__class__.__name__] = {"status": result.status, "errors": result.errors}
            if not result:
                overall_status = False
        

        # Validate or sanitize file paths
        val_status_path = Path(self.config.get_config().val_status).resolve(strict=False)
        root_dir_path = Path(self.config.get_config().root_dir).resolve(strict=False)
        
        # Save results to a file
        try:
            with open(val_status_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            logger.info(f"Validation results saved to {val_status_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")


        # Check and Drop columns with unique values
        self.check_cardinality(data)

        # Save the data to a parquet file only if the validation passed
        if overall_status:
            try:
                output_path = str(root_dir_path / 'validated_data.parquet')
                with open(output_path, 'wb') as f:
                    data.to_parquet(f, index=False)
                logger.info(f"Validated data saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save validated data: {e}")
        else:
            logger.warning(f"Data validation failed. Check {val_status_path} for more details")
        
        return overall_status