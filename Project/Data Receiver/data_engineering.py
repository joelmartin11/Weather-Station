#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration & Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
ENV_PATH = os.path.join(SCRIPT_DIR, ".env.weather")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("CITY", "Chennai")

CSV_PATH = os.path.join(SCRIPT_DIR, "sensor_data.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "processed_features.csv")

# --- Load and Validate Local Sensor Data ---
print(f"Reading local sensor data from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_PATH}. Run receive_data.py first!")
    exit()

# Normalize columns: strip whitespace and lowercase
# This turns "Air Quality" into "air_quality" or "temperature" into "temperature"
df.columns = [c.strip().lower() for c in df.columns]

# Map your specific CSV columns to standard variables
# Based on your provided CSV: temperature, humidity, soil_moisture, air_quality_raw, wind_speed
col_map = {
    "temp": "temperature",
    "hum": "humidity",
    "soil": "soil_moisture",
    "air": "air_quality_raw",
    "wind": "wind_speed"
}

# Check for required columns
if "temperature" not in df.columns or "humidity" not in df.columns:
    print(f"Error: CSV is missing 'temperature' or 'humidity' columns.")
    print(f"Found columns: {list(df.columns)}")
    exit()

# --- Calculate Local Stats ---
# Aggregating the sensor session into single data points
temp_max = df["temperature"].max()
temp_min = df["temperature"].min()
temp_mean = df["temperature"].mean()
temp_range = temp_max - temp_min
humidity_mean = df["humidity"].mean()

# Handle optional sensors (fill with NaN if missing)
soil_moisture = df["soil_moisture"].mean() if "soil_moisture" in df.columns else np.nan
# Your CSV has 'air_quality_raw', so we check for that
aqi_local = df["air_quality_raw"].mean() if "air_quality_raw" in df.columns else np.nan
# Your CSV has 'wind_speed', so we check for that
wind_speed_local_max = df["wind_speed"].max() if "wind_speed" in df.columns else np.nan

# Pressure isn't in your CSV, so we default to NaN
pressure_local = df["pressure"].mean() if "pressure" in df.columns else np.nan

# --- Fetch OpenWeather Data ---
print(f"Fetching API data for {CITY}...")
if not API_KEY:
    print("Error: API Key missing. Check your .env.weather file.")
    exit()

try:
    cur_url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    fore_url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric&cnt=8"

    cur_res = requests.get(cur_url, timeout=10).json()
    fore_res = requests.get(fore_url, timeout=10).json()

    if cur_res.get("cod") != 200:
        raise RuntimeError(f"API Error: {cur_res.get('message')}")

except Exception as e:
    print(f"API Request Failed: {e}")
    exit()

# --- Extract API Features ---
main_data = cur_res["main"]
wind_data = cur_res.get("wind", {})
clouds = cur_res.get("clouds", {}).get("all", np.nan)

pressure_api = main_data.get("pressure", np.nan)
humidity_api = main_data.get("humidity", np.nan)
feels_like_api = main_data.get("feels_like", np.nan)
wind_speed_api = wind_data.get("speed", 0.0) * 3.6  # m/s -> km/h
wind_direction = wind_data.get("deg", np.nan)
visibility_km = cur_res.get("visibility", np.nan) / 1000
precipitation_sum = sum(item.get("rain", {}).get("3h", 0.0) for item in fore_res.get("list", []))

# --- Feature Engineering ---
humidity_index = (precipitation_sum + 0.1) / (wind_speed_api + 0.1)
comfort_index = (humidity_api / 100) * feels_like_api
dew_point_api = main_data.get("temp_min", np.nan) + (humidity_api / 100) * (
    main_data.get("temp_max", np.nan) - main_data.get("temp_min", np.nan)
)
month = datetime.now().month

# --- Combine Everything ---
features = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "city": CITY,
    "month": month,

    # Model Features
    "temperature_2m_max": round(temp_max, 2),
    "temperature_2m_min": round(temp_min, 2),
    "temp_range": round(temp_range, 2),
    "wind_speed_10m_max": round(wind_speed_local_max if not np.isnan(wind_speed_local_max) else wind_speed_api, 2),
    "precipitation_sum": round(precipitation_sum, 2),
    "humidity_index": round(humidity_index, 4),

    # Dashboard Features
    "temperature_mean": round(temp_mean, 2),
    "humidity_local_mean": round(humidity_mean, 2),
    "pressure_local": round(pressure_local, 2),
    "pressure_api": round(pressure_api, 2),
    "humidity_api": round(humidity_api, 2),
    "feels_like_api": round(feels_like_api, 2),
    "dew_point_api": round(dew_point_api, 2),
    "visibility_km": round(visibility_km, 2),
    "cloud_coverage_%": clouds,
    "wind_speed_api": round(wind_speed_api, 2),
    "wind_direction_deg": wind_direction,
    "comfort_index": round(comfort_index, 2),
    "soil_moisture_%": round(soil_moisture, 2),
    "aqi_local": round(aqi_local, 2)
}

# --- Save to CSV ---
pd.DataFrame([features]).to_csv(OUTPUT_PATH, index=False)
print(f"✅ Processed features saved to: {OUTPUT_PATH}")
print("\n--- Feature Summary ---")
for key, value in features.items():
    print(f"{key}: {value}")