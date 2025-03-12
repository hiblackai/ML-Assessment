## How to Run This Project  

I have created this project in a very modular and simple way. Everything is in one place.  

### Folder Structure:  

- **data/**  
  - Contains two subfolders:  
    1. **raw/** – This folder must have a file named `dataset.csv` to train the model.  
    2. **processed/** – This folder will contain all materials generated after data preprocessing. For example, it will contain an image file showing the correlation between features and the target variable.  

- **models/**  
  - This folder stores all trained machine learning models.  

- **scripts/**  
  - This folder contains all the scripts related to machine learning model training.  

### Steps to Run the Project:  

#### **Step 1:** Data Preprocessing  
Run the following command:  

```bash
python scripts/data_preprocessing.py
```
- This script loads the data, preprocesses it, and saves `train.csv` and `test.csv` into the `data/processed` folder.  
- It also saves the encoder and scaler, ensuring that the deployment data is preprocessed with the same consistency.  

#### **Step 2:** Model Training  
Run the following command:  

```bash
python scripts/train_model.py
```
- This script trains an XGBoost regression model.  
- All classes are designed so that parameters can be modified as needed. If not changed, they will work with default settings.  
- After execution, the trained model is saved in the `models/` folder.  

#### **Step 3:** Model Evaluation  
Run the following command:  

```bash
python scripts/evaluate_model.py
```
- This script calculates and displays the RMSE, MAE, and R² scores.  
- You can modify the model name in the script, but by default, it uses the last trained model.  

#### **Step 4:** Hyperparameter Optimization  
Run the following command:  

```bash
python scripts/optimize_model.py
```
- This script starts the hyperparameter tuning process to find the best parameters.  
- After optimization, it trains the model using the best parameters and saves it in the `models/` folder as `best_model.pkl`.  
- The optimized model can be evaluated using `evaluate_model.py`.  

---

### **Model Deployment**  

There are two important files in the root directory:  

- `app.py` – Runs a Flask application that serves both a user interface and an API.  
  - Run the following command to start the app:  
    ```bash
    python app.py
    ```  

- `test_api.py` – Demonstrates how to use the API.  
  - The API accepts POST requests with input in JSON format.  

I have deployed this project on Render. You can check it out here:  
🔗 **[Live App](https://temp-3c37.onrender.com/)**  

**Note:** The application is hosted on a free Render account, so it may take 1-2 minutes to load. If the link is not working, it might be due to inactivity or resource limitations on the free tier.  

---

### **Final Thoughts**  

I made this project with love and dedication. I tried to cover all bonus points. However, I missed version control using MLflow, as I remembered it after completing everything. I do have experience with MLflow and could easily implement it if needed.  

This project was simple for me, as I usually work with transformer-based and computer vision models. I have trained several models currently running in production.  

Thank you! I hope you like this assessment. 😊