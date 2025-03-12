import requests
import json

# API URL
url = "https://temp-3c37.onrender.com/predict"

# Sample input data (same structure as frontend form)
data = {
    "area_type": "Super built-up Area",
    "size": "3 BHK",
    "bath": 2,
    "balcony": 1,
    "total_sqft": "1056",
    "model": "xgboost_model"   
}

# Send request
response = requests.post(url, json=data)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
