import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"
#LOCAL_CITIBIKE_DATA_PATH = Path("notebooks/data/citibike")
LOCAL_CITIBIKE_DATA_PATH = PARENT_DIR / "data" / "raw" 


# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "time_series_6h_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "citi_bike_ds"
FEATURE_VIEW_VERSION = 1


MODEL_NAME = "citybike_lgb_model"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "citi_bike_prediction"
FEATURE_GROUP_MODEL_PREDICTION_VERSION = 2    

