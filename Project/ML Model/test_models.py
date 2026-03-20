#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define File Paths ---
# Just like the training script, we need to know where we are relative to the project root.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODEL_DIR)
BASE_DIR = os.path.dirname(PROJECT_DIR)

# Build the paths dynamically so this runs on your PC and the Pi without changes.
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "india_2000_2024_daily_weather.csv")
CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "weather_class_model.pkl")
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, "temp_forecast_model.pkl")

print("Loading and processing data...")
df = pd.read_csv(DATA_PATH)

# We need to apply the EXACT same cleaning and feature engineering as training.
# If the inputs look different, the models will fail.
df = df.dropna(subset=['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'weather_code'])
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['humidity_index'] = (df['precipitation_sum'] + 0.1) / (df['wind_speed_10m_max'] + 0.1)
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year

# Re-define the simplifier function
def simplify_weather(code):
    if code in [0, 1, 2]: return 0       # Clear/Cloudy
    elif code == 3: return 1             # Overcast
    elif code in [51, 53, 55]: return 2  # Drizzle
    elif code in [61, 63, 65]: return 3  # Rain
    else: return 4                       # Extreme

df['weather_simplified'] = df['weather_code'].apply(simplify_weather)

# --- Load the Pre-trained Models ---
print("Loading models from disk...")
if not os.path.exists(CLASS_MODEL_PATH) or not os.path.exists(TEMP_MODEL_PATH):
    print("Error: One or both model files are missing. Did you run train_models.py?")
    exit()

# Load the bundles containing the models and their specific scalers
class_bundle = joblib.load(CLASS_MODEL_PATH)
reg_bundle = joblib.load(TEMP_MODEL_PATH)
class_model, class_scaler = class_bundle['model'], class_bundle['scaler']
reg_model, reg_scaler = reg_bundle['model'], reg_bundle['scaler']

# --- Evaluate Classification Model ---
print("\n--- Testing Weather Classification Model ---")
class_features = [
    'temperature_2m_max', 'temperature_2m_min', 'temp_range',
    'wind_speed_10m_max', 'precipitation_sum', 'humidity_index', 'month'
]
X_c = df[class_features]
y_c = df['weather_simplified']

# Crucial Step: We recreate the exact random split using seed 42.
# This ensures we are testing on the 20% of data the model has NEVER seen before.
_, X_test_c, _, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

# Scale the test data using the scaler saved during training
X_test_c_scaled = class_scaler.transform(X_test_c)
y_pred_c = class_model.predict(X_test_c_scaled)

print("Classification Report:")
print(classification_report(y_test_c, y_pred_c))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, cmap="Blues", fmt="d")
plt.title("Weather Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
print("Displaying Confusion Matrix... (Close the window to continue)")
plt.show()

# --- Evaluate Regression Model ---
print("\n--- Testing Temperature Regression Model ---")
# Create the target variable again
df['next_day_temp_max'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df = df.dropna(subset=['next_day_temp_max'])

# Filter for the time-based test set (2023 onwards)
test_df_r = df[df['year'] >= 2023]
reg_features = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'wind_speed_10m_max', 'humidity_index', 'month', 'temp_range'
]
X_test_r, y_test_r = test_df_r[reg_features], test_df_r['next_day_temp_max']

# Scale and predict
X_test_r_scaled = reg_scaler.transform(X_test_r)
y_pred_r = reg_model.predict(X_test_r_scaled)

print("Regression Test Results (2023+ Data):")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_r, y_pred_r):.3f}")
print(f"R-squared (R2): {r2_score(y_test_r, y_pred_r):.3f}")
print("\nEvaluation complete.")