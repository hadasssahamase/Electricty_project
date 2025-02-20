from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

# Data Ingestuion config 
@dataclass
class DataIngestionConfig:

    config_data: dict   

# Data Validation config
@dataclass
class DataValidationConfig:
    """
    Configuration for data validation process.

    """
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]
    profile_report_path: str  # Add path for the profile report
    
    
# Data Transformation
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    random_state: frozenset
    target_col: frozenset
    numerical_cols: List[str]
    categorical_cols:List[str]
    
# Model Trainer
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]
    project_name: str
    random_state: int
    number_of_splits: int  
    
         
# Model validation 
@dataclass
class ModelValidationConfig:
    root_dir: Path
    val_features_path: Path
    val_targets_path: Path
    model_path: Path  # Path to the PRE-TRAINED model
    project_name: str
    random_state: int
    
# Model Evaluation
@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_features_path: Path
    test_targets_path: Path
    model_path: Path
    project_name: str
    random_state: int    