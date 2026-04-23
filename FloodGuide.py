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

# Separate in-memory state for each bridge
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
# GET WEATHER FORECAST FROM OPENWEATHERMAP
# Returns: temp, humidity, next 3h rain,
#          next 6h rain (2 forecast periods)
# =========================================
def get_weather_data(lat, lng):
    result = {
        "temp":           28.0,
        "humidity":       70,
        "forecast_3h":    0.0,   # predicted rain in next 3 hours
        "forecast_6h":    0.0,   # predicted rain in next 6 hours
    }
    try:
        # Current weather
        url_now = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lng}&appid={API_KEY}&units=metric"
        )
        r = requests.get(url_now, timeout=5).json()
        result["temp"]     = r["main"]["temp"]
        result["humidity"] = r["main"]["humidity"]
    except Exception:
        pass

    try:
        # Forecast — next 6 hours (cnt=2 gives two 3-hour periods)
        url_fc = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lng}&appid={API_KEY}&units=metric&cnt=2"
        )
        fc = requests.get(url_fc, timeout=5).json()
        periods = fc.get("list", [])
        if len(periods) >= 1:
            result["forecast_3h"] = periods[0].get("rain", {}).get("3h", 0.0)
        if len(periods) >= 2:
            result["forecast_6h"] = (
                periods[0].get("rain", {}).get("3h", 0.0) +
                periods[1].get("rain", {}).get("3h", 0.0)
            )
    except Exception:
        pass

    return result

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
# CORE PREDICTION FUNCTION
# Takes a feature dict, returns risk label
# =========================================
def run_prediction(features, clf):
    """
    features = {
        total_rain, water_level, flow_rate,
        rain_3h, rise_rate, month
    }
    Returns: (risk_label, color, confidence)
    """
    if clf is None:
        return rule_based_predict({
            "rainfall":    features["total_rain"],
            "water_level": features["water_level"],
            "flow_rate":   features["flow_rate"],
            "rain_3h":     features["rain_3h"],
            "rise_rate":   features["rise_rate"]
        })

    df = pd.DataFrame([features])
    try:
        pred          = clf.predict(df)[0]
        probabilities = clf.predict_proba(df)[0]
        confidence    = round(max(probabilities) * 100, 2)
    except Exception as e:
        print(f"⚠️ ML prediction error: {e} — rule-based fallback")
        return rule_based_predict({
            "rainfall":    features["total_rain"],
            "water_level": features["water_level"],
            "flow_rate":   features["flow_rate"],
            "rain_3h":     features["rain_3h"],
            "rise_rate":   features["rise_rate"]
        })

    if pred == 0:   return "Low",      "green",  confidence
    elif pred == 1: return "Moderate", "orange", confidence
    else:           return "High",     "red",    confidence

# =========================================
# PREDICTION ENGINE
# Computes CURRENT risk + PREDICTED risk
# (current + forecast rainfall scenario)
# =========================================
def get_predictions(device_id, forecast_3h, forecast_6h, clf):
    sensor  = latest_sensor_data[device_id].copy()
    history = get_recent_readings(device_id, 6)

    # Compute rolling features from actual history
    if len(history) >= 2:
        rain_values  = [r["total_rain"]  for r in history]
        water_values = [r["water_level"] for r in history]
        rain_3h   = sum(rain_values[-3:]) if len(rain_values) >= 3 else sum(rain_values)
        rise_rate = water_values[-1] - water_values[-2]
    else:
        rain_3h   = sensor["rainfall"]
        rise_rate = 0.0

    # Store computed features for popup display
    latest_sensor_data[device_id]["rain_3h"]   = rain_3h
    latest_sensor_data[device_id]["rise_rate"] = rise_rate

    month = datetime.now().month

    # ── CURRENT RISK ──
    # Based on what sensors are reading right now
    current_features = {
        "total_rain":  sensor["rainfall"],
        "water_level": sensor["water_level"],
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     rain_3h,
        "rise_rate":   rise_rate,
        "month":       month
    }
    current_risk, current_color, current_conf = run_prediction(current_features, clf)

    # ── PREDICTED RISK (3 hours from now) ──
    # Simulates what the sensor will look like in 3h
    # by adding forecast rainfall to current accumulation.
    # This is the actual prediction capability.
    predicted_rain_3h = rain_3h + forecast_3h   # current accumulation + incoming rain
    predicted_wl      = sensor["water_level"] + max(0, forecast_3h * 0.4)  # estimated rise

    predicted_features = {
        "total_rain":  sensor["rainfall"] + forecast_3h,
        "water_level": predicted_wl,
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     predicted_rain_3h,
        "rise_rate":   rise_rate + (forecast_3h * 0.1),
        "month":       month
    }
    pred_risk, pred_color, pred_conf = run_prediction(predicted_features, clf)

    # ── PREDICTED RISK (6 hours from now) ──
    predicted_rain_6h = rain_3h + forecast_6h
    predicted_wl_6h   = sensor["water_level"] + max(0, forecast_6h * 0.4)

    predicted_features_6h = {
        "total_rain":  sensor["rainfall"] + forecast_6h,
        "water_level": predicted_wl_6h,
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     predicted_rain_6h,
        "rise_rate":   rise_rate + (forecast_6h * 0.1),
        "month":       month
    }
    pred_risk_6h, pred_color_6h, pred_conf_6h = run_prediction(predicted_features_6h, clf)

    return {
        "current_risk":  current_risk,
        "current_color": current_color,
        "current_conf":  current_conf,
        "pred_risk_3h":  pred_risk,
        "pred_color_3h": pred_color,
        "pred_conf_3h":  pred_conf,
        "pred_risk_6h":  pred_risk_6h,
        "pred_color_6h": pred_color_6h,
        "pred_conf_6h":  pred_conf_6h,
        "rain_3h":       rain_3h,
        "rise_rate":     rise_rate,
        "forecast_3h":   forecast_3h,
        "forecast_6h":   forecast_6h,
    }

# =========================================
# TREND ARROW HELPER
# =========================================
def trend_arrow(current, predicted):
    levels = {"Low": 0, "Moderate": 1, "High": 2}
    c = levels.get(current, 0)
    p = levels.get(predicted, 0)
    if p > c:   return "↑ Rising",   "#ef4444"
    elif p < c: return "↓ Improving","#22c55e"
    else:       return "→ Stable",   "#94a3b8"

# =========================================
# SUBSIDE TIME ESTIMATOR
# Estimates how long flooding will last
# based on risk level, rainfall, water level,
# and drainage capacity of Mandaue City.
#
# Based on:
# - PAGASA flood bulletin guidelines
# - Mandaue City DRRMO average subside times
# - Urban drainage capacity research (PH)
# =========================================
SUBSIDE_MODEL_PATH = "model_subside.pkl"

def load_subside_model():
    if os.path.exists(SUBSIDE_MODEL_PATH):
        try:
            reg = joblib.load(SUBSIDE_MODEL_PATH)
            return reg
        except Exception:
            return None
    return None

def estimate_subside_time(risk_level, sensor_data, preds):
    """
    Returns (display_text, description, color)
    Always returns a human-readable string, never a raw 0.
    Uses ML regression if available, otherwise rule-based.
    Source: PAGASA guidelines + Mandaue City DRRMO records.
    """
    water_level = float(sensor_data.get("water_level") or 0)
    rain_3h     = float(preds.get("rain_3h")     or 0)
    forecast_3h = float(preds.get("forecast_3h") or 0)
    rise_rate   = float(preds.get("rise_rate")   or 0)
    rainfall    = float(sensor_data.get("rainfall") or 0)

    # Try ML regression model first
    reg = load_subside_model()
    if reg is not None:
        try:
            X_sub = pd.DataFrame([{
                "total_rain":  rainfall,
                "water_level": water_level,
                "flow_rate":   float(sensor_data.get("flow_rate") or 0),
                "rain_3h":     rain_3h,
                "rise_rate":   rise_rate,
                "month":       datetime.now().month
            }])
            predicted_hours = float(reg.predict(X_sub)[0])
            if 0.5 <= predicted_hours <= 48:
                h_min = round(predicted_hours, 1)
                h_max = round(predicted_hours + (1 if predicted_hours < 3 else 2), 1)
                if predicted_hours < 2:
                    return f"{h_min}\u2013{h_max} hrs", "Quick drainage expected", "#22c55e"
                elif predicted_hours < 6:
                    return f"{h_min}\u2013{h_max} hrs", "Moderate drainage time", "#f59e0b"
                else:
                    return f"{h_min}\u2013{h_max} hrs", "Extended flooding expected", "#ef4444"
        except Exception as e:
            print(f"⚠️ Subside ML error: {e}")

    # Rule-based — based on PAGASA thresholds and Mandaue DRRMO data
    if risk_level == "Low":
        if rain_3h < 10 and forecast_3h < 5:
            return "None", "No flooding expected under current conditions", "#22c55e"
        elif rain_3h < 20:
            return "< 1 hour", "Minor surface water — drains quickly", "#22c55e"
        else:
            return "1\u20132 hours", "Light flooding — clears within 1\u20132 hours", "#22c55e"

    elif risk_level == "Moderate":
        if forecast_3h > 15:
            return "3\u20136 hours", "More rain incoming — flooding may extend 3\u20136 hours", "#f59e0b"
        elif water_level > 100:
            return "2\u20134 hours", "Elevated water level — expect 2\u20134 hours to drain", "#f59e0b"
        elif rise_rate > 3:
            return "2\u20135 hours", "Water still rising — may take 2\u20135 hours to subside", "#f59e0b"
        else:
            return "1\u20133 hours", "Moderate flooding — typically clears in 1\u20133 hours", "#f59e0b"

    else:
        if forecast_3h > 30:
            return "6\u201312 hours", "Heavy rain forecast — flooding may persist 6\u201312 hours", "#ef4444"
        elif water_level > 150:
            return "4\u20138 hours", "Severe inundation — roads may be blocked 4\u20138 hours", "#ef4444"
        elif rise_rate > 5:
            return "5\u201310 hours", "Rapidly rising water — extended flooding 5\u201310 hours", "#ef4444"
        else:
            return "3\u20137 hours", "Significant flooding — typically subsides in 3\u20137 hours", "#ef4444"


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

    device_id   = data.get("deviceId", "bridge1")
    water_level = float(data.get("waterLevel", 0))
    rainfall    = float(data.get("totalRain",  0))
    flow_rate   = float(data.get("flowRate",   0))

    if device_id not in latest_sensor_data:
        device_id = "bridge1"

    latest_sensor_data[device_id]["water_level"] = water_level
    latest_sensor_data[device_id]["rainfall"]    = rainfall
    latest_sensor_data[device_id]["flow_rate"]   = flow_rate

    conn   = None
    cursor = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
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
        device_id = loc["id"]

        # Get weather: temp, humidity, forecast rain
        weather      = get_weather_data(loc["lat"], loc["lng"])
        temp         = weather["temp"]
        humidity     = weather["humidity"]
        forecast_3h  = weather["forecast_3h"]
        forecast_6h  = weather["forecast_6h"]

        # Run both current + prediction
        preds = get_predictions(device_id, forecast_3h, forecast_6h, clf)

        # Trend arrow
        trend_text, trend_color = trend_arrow(
            preds["current_risk"], preds["pred_risk_3h"]
        )

        # Subside time estimate
        sensor = latest_sensor_data[device_id]
        sub_display, sub_desc, sub_color = estimate_subside_time(
            preds["pred_risk_3h"], sensor, preds
        )

        subside_display = f'<span style="color:{sub_color};font-weight:700">{sub_display}</span>'

        # Forecast label colors
        def fc_label(mm):
            if mm >= 10:  return f'<span style="color:#ef4444;font-weight:700">{mm:.2f} mm ⚠️ Heavy</span>'
            elif mm >= 2.5: return f'<span style="color:#f59e0b;font-weight:700">{mm:.2f} mm ⚠️ Moderate</span>'
            else:         return f'<span style="color:#22c55e">{mm:.2f} mm (Light/None)</span>'

        popup = f"""
        <div style="font-size:13px; line-height:1.6;">
            <b style="font-size:15px">{loc['name']}</b><br>
            <span style="font-size:11px;color:#64748b">Sensor: {device_id}</span><br><br>

            <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px">
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:4px">📍 Current Risk</div>
                <b style="font-size:18px;color:{preds['current_color']}">
                    {preds['current_risk']} Risk
                </b>
                <span style="font-size:11px;color:#64748b;margin-left:6px">
                    {preds['current_conf']}% confidence
                </span>
            </div>

            <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                        border-left:3px solid {preds['pred_color_3h']}">
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:4px">🔮 Predicted — Next 3 Hours</div>
                <b style="font-size:16px;color:{preds['pred_color_3h']}">
                    {preds['pred_risk_3h']} Risk
                </b>
                <span style="font-size:11px;color:#64748b;margin-left:6px">
                    {preds['pred_conf_3h']}% confidence
                </span><br>
                <span style="font-size:11px;color:{trend_color}">
                    {trend_text}
                </span>
            </div>

            <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                        border-left:3px solid {preds['pred_color_6h']}">
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:4px">🔮 Predicted — Next 6 Hours</div>
                <b style="font-size:14px;color:{preds['pred_color_6h']}">
                    {preds['pred_risk_6h']} Risk
                </b>
                <span style="font-size:11px;color:#64748b;margin-left:6px">
                    {preds['pred_conf_6h']}% confidence
                </span>
            </div>

            <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                        border-left:3px solid {sub_color}">
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:4px">⏱ Estimated Flood Subside Time</div>
                <b style="font-size:15px">{subside_display}</b><br>
                <span style="font-size:11px;color:#94a3b8">{sub_desc}</span>
            </div>

            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">
                <b style="color:#e2e8f0">── Sensor Readings ──</b><br>
                🌧 Rainfall (now): {sensor['rainfall']:.2f} mm<br>
                🌧 Rainfall (3h):  {preds['rain_3h']:.2f} mm<br>
                📏 Water Level:    {sensor['water_level']:.2f} cm<br>
                📈 Rise Rate:      {preds['rise_rate']:.2f} cm/reading<br>
                💧 Flow Rate:      {sensor['flow_rate']:.2f} L
            </div>

            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">
                <b style="color:#e2e8f0">── Weather Forecast ──</b><br>
                🔮 Rain (next 3h): {fc_label(forecast_3h)}<br>
                🔮 Rain (next 6h): {fc_label(forecast_6h)}<br>
                🌡 Temperature:    {temp:.1f} °C<br>
                💧 Humidity:       {humidity}%
            </div>

            <div style="font-size:10px;color:#475569;margin-top:6px">
                🕒 Updated: {timestamp}<br>
                🌲 Prediction: Random Forest + OpenWeatherMap Forecast
            </div>
        </div>
        """

        # Map marker uses the PREDICTED 3h color so it warns BEFORE flood happens
        marker_color = preds["pred_color_3h"]

        data_list.append({
            "id":              loc["id"],
            "lat":             loc["lat"],
            "lng":             loc["lng"],
            "risk_color":      marker_color,   # marker shows predicted risk
            "popup_html":      popup,
            "current_risk":    preds["current_risk"],
            "pred_risk_3h":    preds["pred_risk_3h"],
            "confidence":      preds["current_conf"],
            "water_level":     sensor["water_level"],
            "rainfall":        sensor["rainfall"],
            "flow_rate":       sensor["flow_rate"],
            "rise_rate":       preds["rise_rate"],
            "forecast_3h":     forecast_3h,
        })

    return jsonify({"locations": data_list})

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

        cursor.execute(
            "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 20"
        )
        recent = cursor.fetchall()

        cursor.execute("""
            SELECT
                COUNT(*)                                       AS total,
                SUM(CASE WHEN flooded=2 THEN 1 ELSE 0 END)   AS high,
                SUM(CASE WHEN flooded=1 THEN 1 ELSE 0 END)   AS moderate,
                AVG(CASE WHEN device_id='bridge1'
                         THEN water_level END)                AS avg_water_level,
                MAX(total_rain)                               AS max_rain
            FROM sensor_readings
        """)
        stats = cursor.fetchone()
        cursor.close()
        conn.close()

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