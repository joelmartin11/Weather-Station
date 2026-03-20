* esp32_sender.ino
   ESP32 -> TCP client that streams newline-terminated JSON lines to a Raspberry Pi,
   measures sensors every ~5 seconds, and stops automatically after 2 hours.

   IMPORTANT:
   - Set WIFI_PASS to your Wi-Fi password.
   - SERVER_IP set to 192.168.118.80 as requested.
   - SERVER_PORT default 8266 (must match the Pi server).
*/

#include <WiFi.h>
#include <Wire.h>
#include <DHT.h>
#include <RTClib.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define DHT_PIN 18
#define DHT_TYPE DHT22
#define MQ135_PIN 36
#define SOIL_MOISTURE_PIN 34
#define HALL_PIN 17
#define RAIN_PIN 19

DHT dht(DHT_PIN, DHT_TYPE);
RTC_DS3231 rtc;

const char* WIFI_SSID = "Your_Ssid";
const char* WIFI_PASS = "Your_Pass"; // <-- replace with your Wi-Fi password
const char* SERVER_IP  = "xxx.xxx.xxx.xx";   // <-- Raspberry Pi IP (as requested)
const uint16_t SERVER_PORT = 8266;

const float radius = 0.07;     // m (7 cm)
const int magnets = 3;         // magnets on anemometer
const float mmPerPulse = 0.173; // mm per bucket tip

volatile unsigned long hallPulses = 0;
volatile unsigned long rainPulses = 0;
volatile unsigned long lastRainTime = 0;

WiFiClient client;
const unsigned long RUN_DURATION_MS = 2UL * 60UL * 60UL * 1000UL;  // 2 hours

void IRAM_ATTR hallISR() {
  hallPulses++;
}

void IRAM_ATTR rainISR() {
  unsigned long now = millis();
  if (now - lastRainTime > 100) {
    rainPulses++;
    lastRainTime = now;
  }
}

void connectWiFi() {
  Serial.printf("Connecting to WiFi %s ...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
    if (millis() - start > 20000) {
      Serial.println("\nWiFi connect timeout, restarting...");
      ESP.restart();
    }
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());
}

bool connectServer() {
  if (client && client.connected()) return true;
  Serial.printf("Connecting to server %s:%u ...\n", SERVER_IP, SERVER_PORT);
  if (client.connect(SERVER_IP, SERVER_PORT)) {
    Serial.println("Connected to RPi server.");
    // small keepalive could be set here if desired
    return true;
  } else {
    Serial.println("Failed to connect to server.");
    return false;
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);

  Wire.begin(SDA_PIN, SCL_PIN);
  dht.begin();

  if (!rtc.begin()) {
    Serial.println("RTC not found!");
  } else {
    // Comment out after first upload if RTC already set
    rtc.adjust(DateTime(2025, 11, 11, 12, 00, 00));
    Serial.println("RTC initialized.");
  }

  pinMode(HALL_PIN, INPUT_PULLUP);
  pinMode(RAIN_PIN, INPUT_PULLUP);
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(MQ135_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(HALL_PIN), hallISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(RAIN_PIN), rainISR, FALLING);

  connectWiFi();
}

void loop() {
  static unsigned long startMillis = millis();
  unsigned long elapsed = millis() - startMillis;

  if (elapsed >= RUN_DURATION_MS) {
    Serial.println("\n⏰ 2 hours completed. Stopping data transmission...");

    if (client && client.connected()) {
      client.println("{\"command\":\"STOP\"}"); // newline-terminated STOP
      delay(200);                              // give TCP stack time to send
      client.stop();
    }

    WiFi.disconnect(true);
    delay(50);
    Serial.println("WiFi disconnected. Entering deep sleep...");
    esp_deep_sleep_start(); // halts CPU until reset
  }

  if (!connectServer()) {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    delay(2000);
    return;
  }

  unsigned long measureWindow = 5000; // 5 seconds
  hallPulses = 0;
  delay(measureWindow);

  DateTime now = rtc.now();
  char timestamp[32];
  snprintf(timestamp, sizeof(timestamp), "%04d-%02d-%02dT%02d:%02d:%02d",
           now.year(), now.month(), now.day(),
           now.hour(), now.minute(), now.second());

  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  if (isnan(temp)) temp = 0.0;
  if (isnan(hum)) hum = 0.0;

  int soilRaw = analogRead(SOIL_MOISTURE_PIN);
  int soilPercent = map(soilRaw, 4000, 63, 0, 100);
  soilPercent = constrain(soilPercent, 0, 100);

  int airQ = analogRead(MQ135_PIN);

  noInterrupts();
  unsigned long windPulses = hallPulses;
  unsigned long rainCount = rainPulses;
  interrupts();

  float rotations = windPulses / float(magnets);
  float rps = rotations / (measureWindow / 1000.0);
  float windSpeed_m_s = 2.0 * 3.1415926 * radius * rps;
  float windSpeed_kmh = windSpeed_m_s * 3.6;

  float rainTotal_mm = rainCount * mmPerPulse;

  String json = "{";
  json += "\"timestamp\":\"" + String(timestamp) + "\",";
  json += "\"temperature\":" + String(temp, 2) + ",";
  json += "\"humidity\":" + String(hum, 2) + ",";
  json += "\"soil_moisture\":" + String(soilPercent) + ",";
  json += "\"air_quality_raw\":" + String(airQ) + ",";
  json += "\"wind_speed\":" + String(windSpeed_kmh, 2) + ",";
  json += "\"rain_mm\":" + String(rainTotal_mm, 3) + ",";
  json += "\"device\":\"esp32_1\"";
  json += "}";

  client.println(json); // newline-terminated, safer for server readline
  Serial.println("📤 Sent JSON:");
  Serial.println(json);

  delay(150); // allow TCP stack to flush

  // small additional pause to avoid flooding
  delay(500);
}
