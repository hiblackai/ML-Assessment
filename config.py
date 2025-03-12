import os

# Get the absolute path of the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Model parameters
MODEL_NAME = "xgboost"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameter grid 
PARAM_GRID = {
    "xgboost": {
        "n_estimators": [100, 200,1000],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
    },
}
