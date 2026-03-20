#!/usr/bin/env python3
import socket
import json
import csv
import os
import time
import threading
from datetime import datetime
import sys

# Check if we are on Windows for keypress detection
try:
    import msvcrt
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False  # We are likely on the Raspberry Pi (Linux)

# --- Configuration & Paths ---
# Determine where this script is running to set paths dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# We save the CSV right next to this script, or you can point it to 'Dataset'
OUTPUT_DIR = SCRIPT_DIR
CSV_FILENAME = "sensor_data.csv"
CSV_PATH = os.path.join(OUTPUT_DIR, CSV_FILENAME)

HOST = "0.0.0.0"          # Listen on all network interfaces
PORT = 8266               # Must match the ESP32's target port
DURATION_HOURS = 1        # Run for this long before auto-stopping

# --- Setup CSV File ---
FIELDS = [
    "timestamp",
    "temperature (°C)",
    "humidity (%)",
    "soil_moisture (%)",
    "air_quality_raw",
    "wind_speed (km/h)"
]

# Ensure the directory exists (redundant if using SCRIPT_DIR, but good practice)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the file with headers if it doesn't exist yet
if not os.path.isfile(CSV_PATH):
    print(f"Creating new data file at: {CSV_PATH}")
    with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow(FIELDS)
else:
    print(f"Appending to existing file at: {CSV_PATH}")

# --- Server Initialization ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Allow reusing the address so we don't get "Address already in use" errors on restart
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"📡 Server listening on {HOST}:{PORT}")
print("⏳ Waiting for ESP32 to connect...")

# This blocks until the ESP32 connects
client_socket, client_addr = server_socket.accept()
print(f"✅ ESP32 connected from {client_addr}")

# --- Stop Logic (Threaded) ---
stop_flag = False

def check_stop_condition():
    """
    Background thread to handle manual stopping.
    On Windows: Press any key.
    On Linux/Pi: Use Ctrl+C (handled by KeyboardInterrupt in main loop).
    """
    global stop_flag
    if IS_WINDOWS:
        print("👉 Press ANY KEY to stop early.\n")
        while not stop_flag:
            if msvcrt.kbhit():  # Windows-specific non-blocking check
                _ = msvcrt.getch()
                print("\n🧭 Manual shutdown requested (Key Press)...")
                stop_flag = True
                break
            time.sleep(0.1)
    else:
        print("👉 Press Ctrl+C to stop early.\n")

# Start the listener thread
key_thread = threading.Thread(target=check_stop_condition, daemon=True)
key_thread.start()

# --- Main Data Loop ---
start_time = time.time()
end_time = start_time + (DURATION_HOURS * 3600)
buffer = ""

try:
    while not stop_flag:
        # Check time limit
        if time.time() >= end_time:
            print("\n🕒 Time limit reached. Stopping...")
            stop_flag = True
            break

        # Receive data (blocking call, but with a timeout if we wanted)
        try:
            data = client_socket.recv(1024)
        except ConnectionResetError:
            print("❌ Connection forcibly closed by remote host.")
            break

        if not data:
            print("❌ Connection closed by ESP32.")
            break

        buffer += data.decode("utf-8", errors='ignore')

        # Process complete lines (JSON packets usually end with \n)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                reading = json.loads(line)
                print(f"📥 Data: {reading}")

                # Timestamp: Use ESP32's if available, else current server time
                ts = reading.get("timestamp", datetime.now().isoformat())

                with open(CSV_PATH, "a", newline='', encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        ts,
                        reading.get("temperature", ""),
                        reading.get("humidity", ""),
                        reading.get("soil_moisture", ""),
                        reading.get("air_quality_raw", ""),
                        reading.get("wind_speed", "")
                    ])

            except json.JSONDecodeError:
                print(f"⚠️ Malformed JSON received: {line}")

except KeyboardInterrupt:
    print("\n🛑 Interrupted manually (Ctrl+C).")
    stop_flag = True

finally:
    # Always try to send a stop signal to the ESP32 before closing
    if client_socket:
        try:
            print("🛑 Sending STOP command to ESP32...")
            client_socket.sendall(b"STOP\n")
        except Exception as e:
            print(f"⚠️ Could not send STOP command: {e}")
        client_socket.close()

    server_socket.close()
    print(f"✅ Server closed. All data saved to {CSV_FILENAME}")