from flask import Flask, render_template, jsonify, request
import requests
import joblib
import pandas as pd
from datetime import datetime
import mysql.connector
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
        port=int(os.getenv("MYSQLPORT", 52603))
    )

# =========================================
# CONFIG
# =========================================
API_KEY = "e2a8ee9c6e8ec237763497022a1309bb"

latest_sensor_data = {
    "water_level": 0.0,
    "rainfall": 0.0,
    "flow_rate": 0.0
}

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge", "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# ML MODELS
# =========================================
clf = None
reg = None
model_lock = threading.Lock()

def load_models():
    global clf, reg
    try:
        with model_lock:
            clf = joblib.load("model_class.pkl")
            reg = joblib.load("model_subside.pkl")
            print(f"[{datetime.now()}] Models loaded successfully")
    except Exception as e:
        print("❌ Error loading models:", e)

# Load models at start
load_models()

# =========================================
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    print("Received from ESP32:", data)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    water_level = float(data.get("waterLevel", 0))
    rainfall = float(data.get("totalRain", 0))
    flow_rate = float(data.get("flowRate", 0))

    latest_sensor_data["water_level"] = water_level
    latest_sensor_data["rainfall"] = rainfall
    latest_sensor_data["flow_rate"] = flow_rate

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO sensor_readings (total_rain, water_level, flow_rate) VALUES (%s, %s, %s)"
        cursor.execute(query, (rainfall, water_level, flow_rate))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Data saved to database")
    except Exception as e:
        print("❌ Database Insert Error:", e)

    return jsonify({"status": "received"})

# =========================================
# ML PREDICTION
# =========================================
def ml_predict_from_sensor(sensor_data, clf_model):
    try:
        df = pd.DataFrame([{
            "total_rain": sensor_data["rainfall"],
            "water_level": sensor_data["water_level"],
            "flow_rate": sensor_data["flow_rate"],
            "rain_3h": 0,
            "rain_6h": 0,
            "rise_rate": 0,
            "month": datetime.now().month
        }])
        with model_lock:
            pred = clf_model.predict(df)[0]
            prob = clf_model.predict_proba(df)[0]
        confidence = round(max(prob) * 100, 2)
        if pred == 0:
            return "Low", "green", confidence
        elif pred == 1:
            return "Moderate", "orange", confidence
        else:
            return "High", "red", confidence
    except Exception as e:
        print("❌ Prediction Error:", e)
        return "Unknown", "gray", 0

# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []

    for loc in MONITORED_SITES:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric"
            r = requests.get(url, timeout=5).json()
            rainfall = r.get("rain", {}).get("1h", 0)
            temp = r["main"]["temp"]
            humidity = r["main"]["humidity"]
        except:
            rainfall = 0
            temp = 28
            humidity = 70

        water_level_cm = latest_sensor_data["water_level"]

        risk, color, confidence = ml_predict_from_sensor(latest_sensor_data, clf)

        popup = f"""
        <div style="font-size:14px;">
            <b>{loc['name']}</b><br><br>
            🌊 <b>Flood Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span><br>
            🎯 <b>Confidence:</b> {confidence}%<br><br>
            🌧 Rainfall: {rainfall:.2f} mm<br>
            📏 Water Level: {water_level_cm:.2f} cm<br>
            🌡 Temperature: {temp:.1f} °C<br>
            💧 Humidity: {humidity}%<br><br>
            🕒 Updated: {timestamp}
        </div>
        """
        data_list.append({
            "id": loc["id"],
            "lat": loc["lat"],
            "lng": loc["lng"],
            "risk_color": color,
            "popup_html": popup
        })

    return jsonify({"locations": data_list})

# =========================================
# RETRAIN ENDPOINT
# =========================================
@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        from train_model import load_sensor_data, prepare_features, train_models

        df = load_sensor_data()
        X, y_class, y_reg = prepare_features(df)

        if X is not None:
            train_models(X, y_class, y_reg)
            return jsonify({"status": "retrained"})
        else:
            return jsonify({"status": "no data"}), 400

    except Exception as e:
        print("❌ Retrain Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
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
    app.run(host="0.0.0.0", port=5000, debug=True)