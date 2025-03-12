import os
import sys
import time
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Append project root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import MODEL_DIR, PROCESSED_DATA_DIR, PARAM_GRID
from train_model import XGBoostModelTrainer

class XGBoostOptimizer:
    def __init__(self, data_path, model_dir, param_grid):
        self.data_path = data_path
        self.model_dir = model_dir
        self.param_grid = param_grid
        self.best_params = None
        self.model = None

    def log(self, message):
        """Print a log message with a timestamp."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def load_data(self):
        """Load the dataset and prepare features and target variable."""
        self.log("Loading data...")
        data = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        initial_rows = len(data)
        
        data.dropna(inplace=True)  # Remove missing values
        removed_rows = initial_rows - len(data)
        
        self.log(f"Data loaded. Removed {removed_rows} rows with NaN values.")
        
        X = data.drop(columns=['price'])  # Features
        y = data['price']  # Target variable
        self.log(f"Data shape after preprocessing: X={X.shape}, y={y.shape}")

        return X, y

    def perform_grid_search(self, X, y):
        """Perform hyperparameter optimization using GridSearchCV."""
        self.log("Starting Grid Search for hyperparameter tuning...")

        model = xgb.XGBRegressor()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=2  # Show detailed output
        )

        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        
        self.log(f"Best parameters found: {self.best_params}")

    def train_best_model(self, X, y):
        """Train the model using the best-found hyperparameters."""
        self.log("Training model with best parameters...")
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(X, y)
        self.log("Model training completed.")

    def save_best_model(self):
        """Save the trained model to the specified directory."""
        self.log("Saving best model...")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, 'best_model.pkl')
        joblib.dump(self.model, model_path)

        self.log(f"Best model saved at: {model_path}")

    def run(self):
        """Execute the full optimization pipeline."""
        self.log("Starting optimization pipeline...")

        X, y = self.load_data()
        self.perform_grid_search(X, y)
        self.train_best_model(X, y)
        self.save_best_model()

        self.log("Optimization process completed successfully.")

# Run script
if __name__ == "__main__":
    optimizer = XGBoostOptimizer(PROCESSED_DATA_DIR, MODEL_DIR, PARAM_GRID['xgboost'])
    optimizer.run()
