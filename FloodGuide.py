from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
from datetime import datetime
import mysql.connector
import joblib
import os
import threading
import time

app = Flask(__name__)

# =========================================
# DATABASE CONNECTION (Railway / Render)
# =========================================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "caboose.proxy.rlwy.net"),
        user=os.getenv("MYSQLUSER", "root"),
        password=os.getenv("MYSQLPASSWORD", "cYHockIgVucbKAkgNvkRjTqsTGngkjvD"),
        database=os.getenv("MYSQLDATABASE", "railway"),
        port=int(os.getenv("MYSQLPORT", 52603)),
        connection_timeout=10
    )

# =========================================
# CONFIG
# =========================================
API_KEY = "e2a8ee9c6e8ec237763497022a1309bb"
MODEL_PATH = "model_class.pkl"

# In-memory latest reading (updated on each ESP32 POST)
latest_sensor_data = {
    "water_level": 0.0,
    "rainfall": 0.0,
    "flow_rate": 0.0
}

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge",   "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# RENDER KEEP-ALIVE (prevents spin-down)
# Render free tier sleeps after 15 min idle.
# UptimeRobot pings /ping every 5 min, but
# this thread also self-pings as a backup.
# =========================================
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

def keep_alive():
    """Ping our own /ping endpoint every 10 minutes so Render doesn't spin down."""
    while True:
        time.sleep(600)  # 10 minutes
        if RENDER_URL:
            try:
                requests.get(f"{RENDER_URL}/ping", timeout=10)
                print("✅ Keep-alive ping sent")
            except Exception as e:
                print(f"⚠️ Keep-alive ping failed: {e}")

@app.route("/ping")
def ping():
    """UptimeRobot + self keep-alive endpoint."""
    return jsonify({"status": "alive", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# =========================================
# LOAD MODEL FUNCTION
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            # Guard: model must know at least 2 classes to be useful
            if hasattr(clf, 'classes_') and len(clf.classes_) < 2:
                print("⚠️ Model only has 1 class — falling back to rule-based prediction.")
                return None
            return clf
        except Exception as e:
            print("❌ Error loading model:", e)
            return None
    else:
        print("❌ model_class.pkl not found!")
        return None

# =========================================
# FETCH LAST N READINGS FROM DB
# Used to compute real rain_3h, rain_6h, rise_rate
# =========================================
def get_recent_readings(n=6):
    """Return last n sensor readings as a list of dicts, oldest first."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT total_rain, water_level, flow_rate FROM sensor_readings "
            "ORDER BY timestamp DESC LIMIT %s", (n,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        rows.reverse()  # oldest first
        return rows
    except Exception as e:
        print(f"⚠️ Could not fetch recent readings: {e}")
        return []

# =========================================
# RULE-BASED FALLBACK PREDICTION
# Used when the ML model is not yet trained
# with multiple classes (e.g. early stage).
# Thresholds based on Mandaue flood history.
# =========================================
def rule_based_predict(sensor_data):
    wl   = sensor_data.get("water_level", 0)
    rain = sensor_data.get("rainfall", 0)
    flow = sensor_data.get("flow_rate", 0)
    r3h  = sensor_data.get("rain_3h", rain)
    rise = sensor_data.get("rise_rate", 0)

    # High risk thresholds
    if wl > 150 or r3h > 50 or (rise > 10 and wl > 100):
        return "High", "red", 85.0
    # Moderate risk
    elif wl > 80 or r3h > 20 or flow > 3.0:
        return "Moderate", "orange", 75.0
    # Low risk
    else:
        return "Low", "green", 90.0

# =========================================
# PREDICTION FUNCTION
# Computes real rolling features from DB
# =========================================
def ml_predict_from_sensor(sensor_data, clf):
    # Fetch recent readings to compute rolling features
    history = get_recent_readings(6)

    if len(history) >= 2:
        rain_values  = [r["total_rain"]  for r in history]
        water_values = [r["water_level"] for r in history]

        rain_3h  = sum(rain_values[-3:])   if len(rain_values)  >= 3 else sum(rain_values)
        rain_6h  = sum(rain_values[-6:])   if len(rain_values)  >= 6 else sum(rain_values)
        rise_rate = water_values[-1] - water_values[-2]
    else:
        # Not enough history yet — use current reading as best guess
        rain_3h   = sensor_data["rainfall"]
        rain_6h   = sensor_data["rainfall"]
        rise_rate = 0.0

    # Attach computed features so rule-based fallback can also use them
    sensor_data["rain_3h"]   = rain_3h
    sensor_data["rain_6h"]   = rain_6h
    sensor_data["rise_rate"] = rise_rate

    # If no valid ML model, use rule-based prediction
    if clf is None:
        return rule_based_predict(sensor_data)

    df = pd.DataFrame([{
        "total_rain":  sensor_data["rainfall"],
        "water_level": sensor_data["water_level"],
        "flow_rate":   sensor_data["flow_rate"],
        "rain_3h":     rain_3h,
        "rain_6h":     rain_6h,
        "rise_rate":   rise_rate,
        "month":       datetime.now().month
    }])

    try:
        pred         = clf.predict(df)[0]
        probabilities = clf.predict_proba(df)[0]
        confidence   = round(max(probabilities) * 100, 2)
    except Exception as e:
        print(f"⚠️ ML prediction error: {e} — using rule-based fallback")
        return rule_based_predict(sensor_data)

    if pred == 0:
        return "Low",      "green",  confidence
    elif pred == 1:
        return "Moderate", "orange", confidence
    else:
        return "High",     "red",    confidence

# =========================================
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    water_level = float(data.get("waterLevel", 0))
    rainfall    = float(data.get("totalRain",  0))
    flow_rate   = float(data.get("flowRate",   0))

    # Update in-memory latest
    latest_sensor_data["water_level"] = water_level
    latest_sensor_data["rainfall"]    = rainfall
    latest_sensor_data["flow_rate"]   = flow_rate

    conn   = None
    cursor = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        query  = """
            INSERT INTO sensor_readings (total_rain, water_level, flow_rate)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (rainfall, water_level, flow_rate))
        conn.commit()
        print(f"✅ DB Insert: rain={rainfall}, water={water_level}, flow={flow_rate}")
    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")
    except Exception as e:
        print(f"❌ Unknown DB Error: {e}")
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()

    print("Received from ESP32:", data)
    return jsonify({"status": "received"})

# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []

    clf = load_model()  # None triggers rule-based fallback — never crashes

    for loc in MONITORED_SITES:
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric"
            )
            r        = requests.get(url, timeout=5).json()
            rainfall = r.get("rain", {}).get("1h", 0)
            temp     = r["main"]["temp"]
            humidity = r["main"]["humidity"]
        except Exception:
            rainfall = 0
            temp     = 28
            humidity = 70

        risk, color, confidence = ml_predict_from_sensor(latest_sensor_data, clf)

        water_level_cm = latest_sensor_data["water_level"]
        rain_3h        = latest_sensor_data.get("rain_3h", 0)
        rise_rate      = latest_sensor_data.get("rise_rate", 0)

        popup = f"""
        <div style="font-size:14px;">
            <b>{loc['name']}</b><br><br>
            🌊 <b>Flood Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span><br>
            🎯 <b>Confidence:</b> {confidence}%<br><br>
            🌧 Rainfall (now): {rainfall:.2f} mm<br>
            🌧 Rainfall (3h): {rain_3h:.2f} mm<br>
            📏 Water Level: {water_level_cm:.2f} cm<br>
            📈 Rise Rate: {rise_rate:.2f} cm/reading<br>
            🌡 Temperature: {temp:.1f} °C<br>
            💧 Humidity: {humidity}%<br><br>
            🕒 Updated: {timestamp}
        </div>
        """

        data_list.append({
            "id":         loc["id"],
            "lat":        loc["lat"],
            "lng":        loc["lng"],
            "risk_color": color,
            "popup_html": popup
        })

    return jsonify({"locations": data_list})

# =========================================
# MAIN PAGE
# =========================================
@app.route("/")
def home():
    return render_template("map.html")

# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    print("FloodGuide Running...")

    # Start keep-alive thread only when not in debug reload subprocess
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        t = threading.Thread(target=keep_alive, daemon=True)
        t.start()

    app.run(host="0.0.0.0", port=5000, debug=True)