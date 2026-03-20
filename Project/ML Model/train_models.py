#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# First, let's figure out where we are in the file system.
# This makes the script portable so it runs on your Windows laptop or the Pi without changing code.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODEL_DIR)
BASE_DIR = os.path.dirname(PROJECT_DIR)

# Construct the paths safely using os.path.join
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "india_2000_2024_daily_weather.csv")
CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "weather_class_model.pkl")
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, "temp_forecast_model.pkl")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Clean up the data. We can't train on rows missing key info.
df = df.dropna(subset=['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'weather_code'])

# Feature Engineering: Extracting some useful signals
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
# A rough 'humidity index' estimation since we lack a direct sensor
df['humidity_index'] = (df['precipitation_sum'] + 0.1) / (df['wind_speed_10m_max'] + 0.1)
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year

# Simplify WMO weather codes into 5 main categories for easier classification
def simplify_weather(code):
    if code in [0, 1, 2]: return 0       # Clear/Cloudy
    elif code == 3: return 1             # Overcast
    elif code in [51, 53, 55]: return 2  # Drizzle
    elif code in [61, 63, 65]: return 3  # Rain
    else: return 4                       # Extreme (Thunder/Snow/Fog)

df['weather_simplified'] = df['weather_code'].apply(simplify_weather)

# --- Setup for Classifier (Weather Condition) ---
# We use a random split here because weather *types* (rain vs sun) are somewhat independent day-to-day
class_features = [
    'temperature_2m_max', 'temperature_2m_min', 'temp_range',
    'wind_speed_10m_max', 'precipitation_sum', 'humidity_index', 'month'
]
X_c = df[class_features]
y_c = df['weather_simplified']

# Stratify ensures we keep the same proportion of rare weather events in both sets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)

# --- Setup for Regressor (Temperature Forecast) ---
# We are predicting TOMORROW'S max temp.
df['next_day_temp_max'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df = df.dropna(subset=['next_day_temp_max'])

# IMPORTANT: For forecasting, we must split by TIME, not randomly.
# We train on the past (pre-2023) and test on the future (2023 onwards).
train_df_r = df[df['year'] < 2023]
test_df_r = df[df['year'] >= 2023]

reg_features = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'wind_speed_10m_max', 'humidity_index', 'month', 'temp_range'
]
X_train_r, y_train_r = train_df_r[reg_features], train_df_r['next_day_temp_max']
X_test_r, y_test_r = test_df_r[reg_features], test_df_r['next_day_temp_max']

print(f"Training samples - Classifier: {X_train_c.shape[0]}, Regressor: {X_train_r.shape[0]}")

# --- Training the Classifier ---
print("\nTraining Weather Classifier...")
scaler_class = StandardScaler()
X_train_c_scaled = scaler_class.fit_transform(X_train_c)
X_test_c_scaled = scaler_class.transform(X_test_c)

# We use XGBoost. It's generally the best performing tree-based model for this data.
xgb_class = XGBClassifier(objective='multi:softprob', num_class=5, eval_metric='mlogloss', random_state=42, n_jobs=-1)

# Grid search to find the sweet spot for hyperparameters
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'n_estimators': [150, 250],
    'subsample': [0.8, 1.0]
}
grid_c = GridSearchCV(xgb_class, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_c.fit(X_train_c_scaled, y_train_c)
best_class_model = grid_c.best_estimator_

print(f"Best Classifier Params: {grid_c.best_params_}")
y_pred_c = best_class_model.predict(X_test_c_scaled)
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(classification_report(y_test_c, y_pred_c))

# Save the model bundle (model + scaler)
os.makedirs(os.path.dirname(CLASS_MODEL_PATH), exist_ok=True)
joblib.dump({'model': best_class_model, 'scaler': scaler_class}, CLASS_MODEL_PATH)
print(f"Saved classifier to {CLASS_MODEL_PATH}")

# --- Training the Regressor ---
print("\nTraining Temperature Regressor...")
scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Use TimeSeriesSplit so we don't accidentally validate on past data using future data
ts_cv = TimeSeriesSplit(n_splits=5)
grid_r = GridSearchCV(xgb_reg, param_grid, cv=ts_cv, n_jobs=-1, verbose=1)
grid_r.fit(X_train_r_scaled, y_train_r)
best_reg_model = grid_r.best_estimator_

y_pred_r = best_reg_model.predict(X_test_r_scaled)
print(f"Best Regressor Params: {grid_r.best_params_}")
print(f"MAE: {mean_absolute_error(y_test_r, y_pred_r):.3f}")
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.3f}")

joblib.dump({'model': best_reg_model, 'scaler': scaler_reg}, TEMP_MODEL_PATH)
print(f"Saved regressor to {TEMP_MODEL_PATH}")
print("\nDone. Models are ready for deployment.")