# 🌦️ IoT Weather Station with Machine Learning

An end-to-end IoT-based weather monitoring and prediction system using ESP32, Raspberry Pi, and Machine Learning.

---

## 📌 Overview

This project collects real-time environmental data using sensors connected to an ESP32, sends the data to a Raspberry Pi for processing, and applies machine learning models trained on historical weather data to generate predictions.

---

## ⚙️ System Architecture

ESP32 Sensors → WiFi (TCP) → Raspberry Pi → Data Processing → ML Models → Predictions

---

## 🔧 Components

### 🛰️ ESP32 (Data Collection)

* DHT22 → Temperature & Humidity
* Soil Moisture Sensor
* MQ135 → Air Quality
* Anemometer → Wind Speed
* Rain Sensor

Data is sent as JSON over WiFi to the Raspberry Pi.

---

### 💻 Raspberry Pi (Backend)

#### Data Receiver (`data_receiver.py`)

* TCP server receiving ESP32 data
* Parses JSON
* Stores data in `sensor_data.csv`

#### Data Engineering (`data_engineering.py`)

* Cleans and processes sensor data
* Integrates OpenWeather API
* Generates `processed_features.csv`

#### Machine Learning

* `train_models.py` → trains models
* `test_models.py` → evaluates models
* `run_predictions.py` → generates predictions

---

## 🧠 Machine Learning

### Models Used

* XGBoost Classifier → Weather condition
* XGBoost Regressor → Temperature forecast

### Key Features

* Temperature range
* Wind speed
* Precipitation
* Humidity index
* Seasonal feature (month)

---

## 📁 Project Structure

```
Weather Station/
│
├── ESP32/
│   └── esp32_sender.ino
│
├── Dataset/
│   └── india_2000_2024_daily_weather.csv
│
└── Project/
    ├── Data Receiver/
    │   ├── data_receiver.py
    │   ├── data_engineering.py
    │   └── .env.weather.example
    │
    ├── ML Model/
    │   ├── train_models.py
    │   ├── test_models.py
    │   ├── run_predictions.py
    │
    └── prediction_results.csv
```

---

## 🔐 Environment Setup

Create:

```
Project/Data Receiver/.env.weather
```

Add:

```
OPENWEATHER_API_KEY=your_api_key_here
CITY=Chennai
```

---

## 🚀 How to Run

### 1. Start Data Receiver (Raspberry Pi)

```
python data_receiver.py
```

---

### 2. Upload ESP32 Code

* Open `ESP32/esp32_sender.ino`
* Set WiFi credentials and server IP
* Upload to ESP32

---

### 3. Process Data

```
python data_engineering.py
```

---

### 4. Train Models (First Time)

```
python train_models.py
```

---

### 5. Run Predictions

```
python run_predictions.py
```

---

## 📊 Outputs

* `sensor_data.csv` → raw sensor data
* `processed_features.csv` → engineered features
* `prediction_results.csv` → final predictions

---

## 🎯 Features

* Real-time IoT data collection
* End-to-end data pipeline
* Machine learning integration
* Modular project design
* Works across Windows & Raspberry Pi

---

## ⚠️ Notes

* `.env.weather` is excluded (contains secrets)
* `.pkl` model files are not included
* Output CSV files are generated dynamically

---

## 🚧 Future Improvements

* Web dashboard for visualization
* Cloud deployment
* Database storage
* Mobile notifications

---

## 👨‍💻 Author

Joel Martin
