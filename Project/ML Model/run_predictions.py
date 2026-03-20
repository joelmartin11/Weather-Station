#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import textwrap

# --- Define Dynamic Paths ---
# We are currently in "ML Model"
ML_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to "Project"
PROJECT_DIR = os.path.dirname(ML_DIR)
# Point to the Data Receiver folder where the CSVs live
DATA_DIR = os.path.join(PROJECT_DIR, "Data Receiver")

# Define model and data paths
CLASS_MODEL_FILE = os.path.join(ML_DIR, "weather_class_model.pkl")
REG_MODEL_FILE = os.path.join(ML_DIR, "temp_forecast_model.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "processed_features.csv")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "prediction_results.csv")

# --- Load Models ---
print("🔹 Loading AI models...")

# Check if models exist first
if not os.path.exists(CLASS_MODEL_FILE) or not os.path.exists(REG_MODEL_FILE):
    print("❌ Error: Model files missing. Did you run train_models.py?")
    exit()

# Suppress warnings (sklearn version mismatches can be noisy but harmless here)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    cls_bundle = joblib.load(CLASS_MODEL_FILE)
    reg_bundle = joblib.load(REG_MODEL_FILE)

# Unpack the bundles (Model + Scaler + Feature List)
cls_model = cls_bundle.get("model", cls_bundle)
reg_model = reg_bundle.get("model", reg_bundle)
cls_scaler = cls_bundle.get("scaler")
reg_scaler = reg_bundle.get("scaler")

# Default features if not saved in bundle (fallback safety)
cls_features = cls_bundle.get("features", [
    "temperature_2m_max", "temperature_2m_min", "temp_range",
    "wind_speed_10m_max", "precipitation_sum", "humidity_index", "month"
])
reg_features = reg_bundle.get("features", [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "wind_speed_10m_max", "humidity_index", "month", "temp_range"
])

# --- Load Latest Sensor Data ---
if not os.path.exists(FEATURES_FILE):
    print(f"❌ Error: {FEATURES_FILE} not found.")
    print("   Run 'process_data.py' first to generate the input features.")
    exit()

print(f"🔹 Reading features from: {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
row = df.iloc[0]

# Helper to extract specific columns for a model
def prepare_input(row, feature_list):
    # .get(f, 0.0) ensures we don't crash if a column is missing, defaulting to 0
    data = {f: row.get(f, 0.0) for f in feature_list}
    return pd.DataFrame([data])

X_cls = prepare_input(row, cls_features)
X_reg = prepare_input(row, reg_features)

# Helper to scale data safely
def scale_data(X, scaler, name):
    if scaler is None:
        print(f"⚠️  Warning: No scaler found for {name}. Using raw data.")
        return X.values
    try:
        return scaler.transform(X)
    except Exception as e:
        print(f"⚠️  Scaling failed for {name}: {e}. Using raw data.")
        return X.values

X_cls_scaled = scale_data(X_cls, cls_scaler, "classifier")
X_reg_scaled = scale_data(X_reg, reg_scaler, "regressor")

# --- Run Predictions ---
print("🔹 Running inference...")

# Classifier Prediction (Weather Type)
if hasattr(cls_model, "predict_proba"):
    probs = cls_model.predict_proba(X_cls_scaled)[0]
    cls_label = int(np.argmax(probs))
else:
    cls_label = int(cls_model.predict(X_cls_scaled)[0])
    probs = None

# Regressor Prediction (Max Temp)
pred_temp = float(reg_model.predict(X_reg_scaled)[0])

# Map codes to human-readable text
weather_labels = {
    0: "Clear / Partly Cloudy ☀️",
    1: "Overcast ☁️",
    2: "Light Drizzle 🌦️",
    3: "Rain 🌧️",
    4: "Stormy / Extreme 🌫️"
}
label_text = weather_labels.get(cls_label, f"Unknown Code ({cls_label})")

# --- Save Results ---
output_data = row.to_dict()
output_data.update({
    "predicted_weather_code": cls_label,
    "predicted_weather_label": label_text,
    "predicted_temp_max": round(pred_temp, 2),
    "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

# Reorder columns to put predictions first (easier to read in Excel)
cols = ["prediction_timestamp", "predicted_weather_label", "predicted_temp_max"] + [c for c in output_data if c not in ["prediction_timestamp", "predicted_weather_label", "predicted_temp_max"]]
pd.DataFrame([output_data], columns=cols).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Results saved to: {OUTPUT_FILE}")

# --- Generate Detailed Human-Readable Report ---
print("\n" + "="*60)
print(f"  🌤️  DETAILED WEATHER FORECAST FOR {output_data.get('city', 'UNKNOWN').upper()}")
print("="*60)

# Extract variables for the story
city = output_data.get("city", "your location")
pred_label = output_data.get("predicted_weather_label")
pred_temp_val = output_data.get("predicted_temp_max")
feels_like = output_data.get('feels_like_api', np.nan)
wind_speed = output_data.get('wind_speed_10m_max', np.nan)
precip_sum = output_data.get('precipitation_sum', 0.0)
humidity = output_data.get('humidity_local_mean', output_data.get('humidity_api', np.nan))
clouds = output_data.get('cloud_coverage_%', np.nan)
visibility = output_data.get('visibility_km', np.nan)
soil = output_data.get('soil_moisture_%', np.nan)
aqi_raw = output_data.get('aqi_local', np.nan)

# Build the Narrative parts
report_parts = []

# 1. The Headline
intro = f"Tomorrow in {city}, the forecast calls for **{pred_label}** with a high of **{pred_temp_val:.1f}°C**."
report_parts.append(intro)

# 2. Temperature & RealFeel (The "Sweat Factor")
if not np.isnan(feels_like):
    diff = feels_like - pred_temp_val
    if diff > 4:
        report_parts.append(f"⚠️ **Heat Advisory**: It's a trap! The high humidity will make it feel much hotter, around **{feels_like:.1f}°C**. Hydrate or you'll regret it.")
    elif diff > 2:
        report_parts.append(f"It will feel a bit sticky, closer to **{feels_like:.1f}°C** due to the humidity.")
    elif diff < -3:
        report_parts.append(f"The wind chill is real tomorrow—it'll feel brisk, like **{feels_like:.1f}°C**. Bring a jacket.")
    else:
        report_parts.append("The temperature will feel pretty true to the forecast.")

# 3. Rain & Umbrella Logic
if precip_sum > 20.0:
    report_parts.append(f"🌧️ **Heavy Rain Alert**: We are expecting significant rainfall (~{precip_sum:.1f} mm). Local flooding is possible; drive slow and stay dry.")
elif precip_sum > 5.0:
    report_parts.append(f"☔ It's definitely an umbrella day. Expect consistent showers (~{precip_sum:.1f} mm) throughout the day.")
elif precip_sum > 0.5:
    report_parts.append(f"🌦️ There's a chance of scattered showers (~{precip_sum:.1f} mm). Keep an umbrella handy just in case.")
else:
    if not np.isnan(clouds):
        if clouds > 80:
            report_parts.append("☁️ It will be a dry but gray, overcast day.")
        elif clouds < 20:
            report_parts.append("☀️ Skies should be mostly clear and sunny. Great for solar projects.")
        else:
            report_parts.append("⛅ Expect a mix of sun and clouds.")
    else:
        report_parts.append("It looks like a dry day ahead.")

# 4. Wind Conditions
if not np.isnan(wind_speed):
    if wind_speed > 40:
        report_parts.append(f"💨 **Gale Warning**: Winds could gust over **{wind_speed:.1f} km/h**. Secure your trash cans and hold onto your hat.")
    elif wind_speed > 20:
        report_parts.append(f"It will be quite breezy with gusts up to **{wind_speed:.1f} km/h**, which might help cool things down.")
    else:
        report_parts.append("Winds will be calm.")

# 5. Visibility & Atmosphere
if not np.isnan(visibility) and visibility < 2.0:
    report_parts.append(f"🌁 **Fog Warning**: Visibility is low (~{visibility:.1f} km). Be careful if you're driving early in the morning.")

# 6. Soil & Gardening Tip (Contextual)
if not np.isnan(soil):
    # Assuming soil moisture sensor returns 0-100%
    if soil < 10 and precip_sum < 1.0:
        report_parts.append("🌱 **Gardening Tip**: Your soil sensors are reading bone dry. Your plants are thirsty—water them!")
    elif soil > 80:
        report_parts.append("🌱 **Gardening Tip**: The soil is saturated. No need to water the garden tomorrow.")

# 7. Air Quality (Raw context)
if not np.isnan(aqi_raw):
    report_parts.append(f"(Sensor Note: Air Quality raw reading is {aqi_raw:.0f}).")

# Join and Print
full_report = " ".join(report_parts)
print("\n" + textwrap.fill(full_report, width=80))

# Show probabilities if available
if probs is not None:
    print("\n--- ML Confidence Scores ---")
    for i, p in enumerate(probs):
        marker = "👈" if i == cls_label else ""
        print(f"  {weather_labels.get(i):<25} : {p*100:5.1f}% {marker}")

print("\n" + "="*60 + "\n")