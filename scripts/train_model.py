import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODEL_DIR, PROCESSED_DATA_DIR
import xgboost as xgb
import pandas as pd
import joblib

class XGBoostModelTrainer:
    def __init__(self, data_path, model_dir, params=None):
        self.data_path = data_path
        self.model_dir = model_dir
        
        # Default hyperparameters
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
        }
        
        # Use custom parameters if provided
        self.params = params if params else self.default_params
        self.model = None

    def load_data(self):
        """Load the dataset and drop rows with NaN values."""
        self.data = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        self.data.dropna(inplace=True)
        print("Data loaded and NaN values dropped.")

    def train_model(self):
        """Train the XGBoost model with specified hyperparameters."""
        X = self.data.drop(columns=['price'])  # Drop the target column to get features
        y = self.data['price']  # Select the target column


        # Initialize the XGBoost Regressor with the provided/custom parameters
        self.model = xgb.XGBRegressor(**self.params)

        # Train the model
        self.model.fit(X, y)
        print("Model training completed.")

    def save_model(self):
        """Save the trained model to the specified directory."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def run(self):
        """Run the entire pipeline: load data, train model, and save model."""
        self.load_data()
        self.train_model()
        self.save_model()

# Example usage:
if __name__ == "__main__": 

    trainer = XGBoostModelTrainer(PROCESSED_DATA_DIR, MODEL_DIR)
    trainer.run()
