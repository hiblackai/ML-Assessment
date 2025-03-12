import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import RAW_DATA_PATH, PROCESSED_DATA_DIR, TEST_SIZE

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# Data Cleaning & Preprocessing
## Handling 'size' column (Extracting numerical value)
df['size'] = df['size'].astype(str).str.extract(r'(\d+)').astype(float)


## Handling 'total_sqft' (convert range to mean)
def convert_total_sqft(value):
    try:
        if '-' in value:
            low, high = value.split('-')
            return (float(low) + float(high)) / 2
        return float(value)
    except ValueError:
        return np.nan  # Invalid values replaced with NaN

df['total_sqft'] = df['total_sqft'].astype(str).apply(convert_total_sqft)

## Handling missing values
df.dropna(inplace=True)

# Encoding categorical variables
cat_features = ['area_type']
# Remove unnecessary spaces from 'area_type'
df['area_type'] = df['area_type'].str.strip().replace(r'\s+', ' ', regex=True)
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = one_hot_encoder.fit_transform(df[cat_features])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(cat_features))

# Merging encoded features
df = df.drop(columns=cat_features)
df = pd.concat([df, encoded_df], axis=1)

# Feature Scaling
scaler = StandardScaler()
numerical_features = ['size', 'bath', 'balcony', 'total_sqft']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Splitting dataset
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Save train and test data
train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)

# Save preprocessing objects
joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, "scaler.pkl"))
joblib.dump(one_hot_encoder, os.path.join(PROCESSED_DATA_DIR, "encoder.pkl"))

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(PROCESSED_DATA_DIR, "correlation_heatmap.png"))
plt.close()

print("Data preprocessing completed. Train & test data saved.")


