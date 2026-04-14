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
# DATABASE CONNECTION (Clever Cloud)
# =========================================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "b7hubkhd0btnnd1gtw6f-mysql.services.clever-cloud.com"),
        user=os.getenv("MYSQLUSER", "ufmtd04ysstwmoml"),
        password=os.getenv("MYSQLPASSWORD", "X0kiVkBuBLacKOxNtUkI"),
        database=os.getenv("MYSQLDATABASE", "b7hubkhd0btnnd1gtw6f"),
        port=int(os.getenv("MYSQLPORT", 3306)),
        connection_timeout=10,
        ssl_disabled=False
    )

# =========================================
# CONFIG
# =========================================
API_KEY    = "e2a8ee9c6e8ec237763497022a1309bb"
MODEL_PATH = "model_class.pkl"

# Separate in-memory state for EACH bridge
latest_sensor_data = {
    "bridge1": {
        "water_level": 0.0,
        "rainfall":    0.0,
        "flow_rate":   0.0,
        "rain_3h":     0.0,
        "rise_rate":   0.0
    },
    "bridge2": {
        "water_level": 0.0,
        "rainfall":    0.0,
        "flow_rate":   0.0,
        "rain_3h":     0.0,
        "rise_rate":   0.0
    }
}

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge",   "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# RENDER KEEP-ALIVE
# =========================================
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

def keep_alive():
    while True:
        time.sleep(600)
        if RENDER_URL:
            try:
                requests.get(f"{RENDER_URL}/ping", timeout=10)
                print("✅ Keep-alive ping sent")
            except Exception as e:
                print(f"⚠️ Keep-alive ping failed: {e}")

@app.route("/ping")
def ping():
    return jsonify({"status": "alive", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            if hasattr(clf, 'classes_') and len(clf.classes_) < 2:
                print("⚠️ Model only has 1 class — using rule-based fallback.")
                return None
            return clf
        except Exception as e:
            print("❌ Error loading model:", e)
            return None
    else:
        print("❌ model_class.pkl not found!")
        return None

# =========================================
# FETCH RECENT READINGS PER DEVICE
# =========================================
def get_recent_readings(device_id, n=6):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT total_rain, water_level, flow_rate FROM sensor_readings "
            "WHERE device_id = %s ORDER BY timestamp DESC LIMIT %s",
            (device_id, n)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        rows.reverse()
        return rows
    except Exception as e:
        print(f"⚠️ Could not fetch readings for {device_id}: {e}")
        return []

# =========================================
# RULE-BASED FALLBACK PREDICTION
# =========================================
def rule_based_predict(sensor_data):
    wl   = sensor_data.get("water_level", 0)
    rain = sensor_data.get("rainfall",    0)
    flow = sensor_data.get("flow_rate",   0)
    r3h  = sensor_data.get("rain_3h",     rain)
    rise = sensor_data.get("rise_rate",   0)

    if wl > 150 or r3h > 50 or (rise > 10 and wl > 100):
        return "High",     "red",    85.0
    elif wl > 80 or r3h > 20 or flow > 3.0:
        return "Moderate", "orange", 75.0
    else:
        return "Low",      "green",  90.0

# =========================================
# PREDICTION FUNCTION (per device)
# =========================================
def ml_predict_from_sensor(device_id, clf):
    sensor_data = latest_sensor_data[device_id].copy()
    history     = get_recent_readings(device_id, 6)

    if len(history) >= 2:
        rain_values  = [r["total_rain"]  for r in history]
        water_values = [r["water_level"] for r in history]
        rain_3h   = sum(rain_values[-3:]) if len(rain_values) >= 3 else sum(rain_values)
        rise_rate = water_values[-1] - water_values[-2]
    else:
        rain_3h   = sensor_data["rainfall"]
        rise_rate = 0.0

    sensor_data["rain_3h"]   = rain_3h
    sensor_data["rise_rate"] = rise_rate

    # Store back so popup can display them
    latest_sensor_data[device_id]["rain_3h"]   = rain_3h
    latest_sensor_data[device_id]["rise_rate"] = rise_rate

    if clf is None:
        return rule_based_predict(sensor_data)

    df = pd.DataFrame([{
        "total_rain":  sensor_data["rainfall"],
        "water_level": sensor_data["water_level"],
        "flow_rate":   sensor_data["flow_rate"],
        "rain_3h":     rain_3h,
        "rise_rate":   rise_rate,
        "month":       datetime.now().month
    }])

    try:
        pred          = clf.predict(df)[0]
        probabilities = clf.predict_proba(df)[0]
        confidence    = round(max(probabilities) * 100, 2)
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
# AUTO-RETRAIN ENDPOINT
# =========================================
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    try:
        print("🔄 Auto-retrain triggered...")
        conn = get_db_connection()
        df   = pd.read_sql(
            "SELECT * FROM sensor_readings ORDER BY timestamp ASC", conn
        )
        conn.close()

        if len(df) < 10:
            return jsonify({"status": "skipped",
                            "reason": f"Only {len(df)} rows — need at least 10"}), 200

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["rain_3h"]   = df["total_rain"].rolling(window=3, min_periods=1).sum()
        df["rise_rate"] = df["water_level"].diff().fillna(0)
        df["month"]     = df["timestamp"].dt.month
        df = df.fillna(0)

        if df["flooded"].nunique() < 2:
            df["flooded"] = df.apply(
                lambda r: 2 if (r.water_level > 150 or r.total_rain > 30)
                         else (1 if (r.water_level > 80 or r.total_rain > 15
                                     or r.flow_rate > 3)
                         else 0), axis=1
            )

        if df["flooded"].nunique() < 2:
            return jsonify({"status": "skipped",
                            "reason": "Still only 1 class after auto-label"}), 200

        from sklearn.ensemble import RandomForestClassifier
        features = ["total_rain", "water_level", "flow_rate",
                    "rain_3h", "rise_rate", "month"]
        X = df[features]
        y = df["flooded"].astype(int)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42
        )
        clf.fit(X, y)
        joblib.dump(clf, MODEL_PATH)

        msg = f"✅ Retrained on {len(df)} rows — classes: {sorted(y.unique().tolist())}"
        print(msg)
        return jsonify({"status": "success", "message": msg, "rows": len(df)}), 200

    except Exception as e:
        print(f"❌ Retrain error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # Read device_id from ESP32 payload
    device_id   = data.get("deviceId", "bridge1")
    water_level = float(data.get("waterLevel", 0))
    rainfall    = float(data.get("totalRain",  0))
    flow_rate   = float(data.get("flowRate",   0))

    # Validate — only allow known devices
    if device_id not in latest_sensor_data:
        print(f"⚠️ Unknown device_id: {device_id} — defaulting to bridge1")
        device_id = "bridge1"

    # Update THIS bridge's in-memory state
    latest_sensor_data[device_id]["water_level"] = water_level
    latest_sensor_data[device_id]["rainfall"]    = rainfall
    latest_sensor_data[device_id]["flow_rate"]   = flow_rate

    conn   = None
    cursor = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        # Save with device_id so each bridge's data is separate
        cursor.execute(
            "INSERT INTO sensor_readings (device_id, total_rain, water_level, flow_rate) "
            "VALUES (%s, %s, %s, %s)",
            (device_id, rainfall, water_level, flow_rate)
        )
        conn.commit()
        print(f"✅ DB Insert [{device_id}]: rain={rainfall}, water={water_level}, flow={flow_rate}")
    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")
    except Exception as e:
        print(f"❌ Unknown DB Error: {e}")
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()

    print(f"Received from ESP32 [{device_id}]:", data)
    return jsonify({"status": "received", "device": device_id})

# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []
    clf       = load_model()

    for loc in MONITORED_SITES:
        device_id = loc["id"]  # "bridge1" or "bridge2"

        # OpenWeatherMap: temperature, humidity, 3h forecast
        temp             = 28
        humidity         = 70
        forecast_rain_3h = 0.0

        try:
            url_weather = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric"
            )
            r        = requests.get(url_weather, timeout=5).json()
            temp     = r["main"]["temp"]
            humidity = r["main"]["humidity"]
        except Exception:
            pass

        try:
            url_forecast = (
                f"https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric&cnt=1"
            )
            f_data           = requests.get(url_forecast, timeout=5).json()
            forecast_rain_3h = f_data["list"][0].get("rain", {}).get("3h", 0.0)
        except Exception:
            forecast_rain_3h = 0.0

        # Get prediction using THIS bridge's own sensor data
        risk, color, confidence = ml_predict_from_sensor(device_id, clf)

        # Read THIS bridge's sensor values
        sensor         = latest_sensor_data[device_id]
        water_level_cm = sensor["water_level"]
        current_rain   = sensor["rainfall"]
        rain_3h        = sensor.get("rain_3h",   0)
        rise_rate      = sensor.get("rise_rate", 0)
        flow_rate      = sensor["flow_rate"]

        # Color-coded forecast label
        if forecast_rain_3h >= 10:
            forecast_label = (f'<span style="color:red;font-weight:bold;">'
                              f'{forecast_rain_3h:.2f} mm ⚠️ Heavy</span>')
        elif forecast_rain_3h >= 2.5:
            forecast_label = (f'<span style="color:orange;font-weight:bold;">'
                              f'{forecast_rain_3h:.2f} mm ⚠️ Moderate</span>')
        else:
            forecast_label = (f'<span style="color:green;">'
                              f'{forecast_rain_3h:.2f} mm (Light/None)</span>')

        popup = f"""
        <div style="font-size:14px;">
            <b>{loc['name']}</b><br>
            <small style="color:gray;">Sensor: {device_id}</small><br><br>
            🌊 <b>Flood Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span><br>
            🎯 <b>Confidence:</b> {confidence}%<br><br>
            <b>── Sensor Readings ──</b><br>
            🌧 <b>Rainfall (now):</b>  {current_rain:.2f} mm<br>
            🌧 <b>Rainfall (3h):</b>   {rain_3h:.2f} mm<br>
            📏 <b>Water Level:</b>     {water_level_cm:.2f} cm<br>
            📈 <b>Rise Rate:</b>       {rise_rate:.2f} cm/reading<br>
            💧 <b>Flow Rate:</b>       {flow_rate:.2f} L<br><br>
            <b>── Weather Forecast ──</b><br>
            🔮 <b>Rainfall (next 3h):</b> {forecast_label}<br>
            🌡 <b>Temperature:</b>     {temp:.1f} °C<br>
            💧 <b>Humidity:</b>        {humidity}%<br><br>
            🕒 <b>Updated:</b> {timestamp}
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
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        t = threading.Thread(target=keep_alive, daemon=True)
        t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)


# =========================================
# DASHBOARD PAGE + API
# =========================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/dashboard-data")
def dashboard_data():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Last 50 readings per bridge
        cursor.execute(
            "SELECT * FROM sensor_readings WHERE device_id = 'bridge1' "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        bridge1 = list(reversed(cursor.fetchall()))

        cursor.execute(
            "SELECT * FROM sensor_readings WHERE device_id = 'bridge2' "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        bridge2 = list(reversed(cursor.fetchall()))

        # Recent 20 rows (all devices)
        cursor.execute(
            "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 20"
        )
        recent = cursor.fetchall()

        # Summary stats
        cursor.execute("""
            SELECT
                COUNT(*)                                          AS total,
                SUM(CASE WHEN flooded = 2 THEN 1 ELSE 0 END)    AS high,
                SUM(CASE WHEN flooded = 1 THEN 1 ELSE 0 END)    AS moderate,
                AVG(CASE WHEN device_id='bridge1'
                         THEN water_level END)                   AS avg_water_level,
                MAX(total_rain)                                  AS max_rain
            FROM sensor_readings
        """)
        stats = cursor.fetchone()

        cursor.close()
        conn.close()

        # Convert timestamps to strings for JSON
        def serialize(rows):
            out = []
            for r in rows:
                row = dict(r)
                if hasattr(row.get('timestamp'), 'isoformat'):
                    row['timestamp'] = row['timestamp'].isoformat()
                out.append(row)
            return out

        return jsonify({
            "bridge1": serialize(bridge1),
            "bridge2": serialize(bridge2),
            "recent":  serialize(recent),
            "stats":   {k: (float(v) if v is not None else 0)
                        for k, v in stats.items()}
        })

    except Exception as e:
        print(f"❌ Dashboard data error: {e}")
        return jsonify({"error": str(e)}), 500