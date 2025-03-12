import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODEL_DIR, PROCESSED_DATA_DIR

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class XGBoostModelEvaluator:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Load the test dataset and drop rows with NaN values."""
        self.data = pd.read_csv(self.data_path + '\\test.csv')
        self.data.dropna(inplace=True)
        print("Test data loaded and NaN values dropped.")

    def load_model(self):
        """Load the trained XGBoost model."""
        self.model = joblib.load(self.model_path)
        print("Model loaded.")

    def prepare_data(self):
        """Prepare the test data for evaluation."""
        # Assuming the last column is the target variable
        self.X_test = self.data.drop(columns=['price'])  # Drop the target column to get features
        self.y_test = self.data['price']  # Select the target column
        print("Data prepared for evaluation.")

    def evaluate_model(self):
        """Evaluate the model and calculate RMSE, MAE, and R²."""
        if self.model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model or test data not loaded. Please ensure data and model are loaded.")

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Display metrics
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R²: {r2}")

        return rmse, mae, r2

    def run(self):
        """Run the entire evaluation pipeline: load data, load model, prepare data, and evaluate."""
        self.load_data()
        self.load_model()
        self.prepare_data()
        self.evaluate_model()

# Example usage:
if __name__ == "__main__":
    evaluator = XGBoostModelEvaluator(PROCESSED_DATA_DIR, os.path.join(MODEL_DIR, 'xgboost_model.pkl')) # Select any custom model 
    evaluator.run()